import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearProjectionHead(nn.Module):
    """Linear projection head that maps vision features to shared embedding space."""

    def __init__(self, in_features: int, out_features: int = 1024):
        super().__init__()
        self.proj_t1 = nn.Linear(in_features, out_features)
        self.proj_t2 = nn.Linear(in_features, out_features)

    def forward_t1(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t1(x), dim=-1)

    def forward_t2(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t2(x), dim=-1)


class TextProjectionHead(nn.Module):
    """Linear projection head that maps text features to shared embedding space."""

    def __init__(self, in_features: int, out_features: int = 1024):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class AttentionPooler(nn.Module):
    """Gated attention pooler for aggregating slice-level embeddings to series-level."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout_p: float = 0.25):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.attn_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        if dropout_p > 0:
            self.attn_a.append(nn.Dropout(dropout_p))
            self.attn_b.append(nn.Dropout(dropout_p))
        self.attn_c = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slices, input_dim)
            pad_mask: (batch, num_slices) boolean, True for valid slices
        Returns:
            (batch, hidden_dim) aggregated representation
        """
        B, S, _ = x.shape
        h = self.phi(x)

        h_flat = h.reshape(B * S, -1)
        a = self.attn_a(h_flat)
        b = self.attn_b(h_flat)
        scores = self.attn_c(a * b).view(B, S)

        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)

        aggregated = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return self.head(aggregated)


class LayerWiseSynthesizer(nn.Module):
    """Layer-wise router that fuses T1 and T2 intermediate representations.

    At each encoder layer, computes fusion weights to combine T1 and T2
    representations via weighted sum, producing series-agnostic embeddings.
    """

    def __init__(self, num_layers: int, embed_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.fusion_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_layers)
        ])

    def get_weight(self, layer_idx: int) -> torch.Tensor:
        return torch.sigmoid(self.fusion_logits[layer_idx])

    def fuse(self, h_t1: torch.Tensor, h_t2: torch.Tensor, layer_idx: int) -> torch.Tensor:
        alpha = self.get_weight(layer_idx)
        return alpha * h_t1 + (1 - alpha) * h_t2


class SynthesizerEncoder(nn.Module):
    """Wrapper that runs two ViT encoders with layer-wise fusion via synthesizer.

    Processes input through both T1 and T2 encoders simultaneously, fusing
    intermediate representations at each layer to produce a unified embedding.

    The forward pass mirrors ``DinoVisionTransformer.forward_features`` but
    applies the synthesizer at every layer:
      1. Both encoders ``prepare_tokens_with_masks`` (patch embed + CLS/storage).
      2. Compute shared RoPE from the spatial resolution.
      3. At each block, run T1 block and T2 block, then fuse.
      4. Apply the T1 encoder's final norm and return the CLS token.
    """

    def __init__(self, t1_encoder: nn.Module, t2_encoder: nn.Module, synthesizer: LayerWiseSynthesizer):
        super().__init__()
        self.t1_encoder = t1_encoder
        self.t2_encoder = t2_encoder
        self.synthesizer = synthesizer

        for param in self.t1_encoder.parameters():
            param.requires_grad = False
        for param in self.t2_encoder.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self):
        return self.t1_encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_t1, (H, W) = self.t1_encoder.prepare_tokens_with_masks(x)
        h_t2, _ = self.t2_encoder.prepare_tokens_with_masks(x)

        t1_rope = self.t1_encoder.rope_embed(H=H, W=W) if self.t1_encoder.rope_embed is not None else None
        t2_rope = self.t2_encoder.rope_embed(H=H, W=W) if self.t2_encoder.rope_embed is not None else None

        for layer_idx, (t1_block, t2_block) in enumerate(
            zip(self.t1_encoder.blocks, self.t2_encoder.blocks)
        ):
            h_t1_out = t1_block(h_t1, t1_rope)
            h_t2_out = t2_block(h_t2, t2_rope)
            h_fused = self.synthesizer.fuse(h_t1_out, h_t2_out, layer_idx)
            h_t1 = h_fused
            h_t2 = h_fused

        x_normed = self.t1_encoder.norm(h_fused)
        cls_token = x_normed[:, 0]
        return cls_token


class CLIPModel(nn.Module):
    """Full CLIP model for SpineFM text-image alignment."""

    def __init__(
        self,
        vision_proj: LinearProjectionHead,
        text_proj: TextProjectionHead,
        t1_pooler: AttentionPooler,
        t2_pooler: AttentionPooler,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.t1_pooler = t1_pooler
        self.t2_pooler = t2_pooler
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_features)

    def encode_t1_series(
        self, slice_embeddings: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.t1_pooler(slice_embeddings, pad_mask)
        return self.vision_proj.forward_t1(pooled)

    def encode_t2_series(
        self, slice_embeddings: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.t2_pooler(slice_embeddings, pad_mask)
        return self.vision_proj.forward_t2(pooled)

    def forward(
        self,
        t1_embeddings: torch.Tensor,
        t1_mask: torch.Tensor,
        t2_embeddings: torch.Tensor,
        t2_mask: torch.Tensor,
        text_features: torch.Tensor,
    ):
        t1_repr = self.encode_t1_series(t1_embeddings, t1_mask)
        t2_repr = self.encode_t2_series(t2_embeddings, t2_mask)
        image_repr = (t1_repr + t2_repr) / 2.0

        text_repr = self.encode_text(text_features)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_repr @ text_repr.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor):
    """Symmetric contrastive loss for CLIP (Eq. 2 in the paper)."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2.0
