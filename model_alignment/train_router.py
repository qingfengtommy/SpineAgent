"""Router/Synthesizer training for SpineFM.

Trains a layer-wise synthesizer module that fuses T1 and T2 encoder
representations to produce series-agnostic embeddings for arbitrary
MRI sequences.

Two modes are supported:

  --mode embedding  (default)
      Works with pre-extracted T1/T2 embeddings.  Learns an embedding-level
      fusion weight and CLIP-aligned attention pooler.  Fast to iterate.

  --mode image
      Loads both DINOv3 encoders and runs raw images through the full
      SynthesizerEncoder with layer-wise fusion.  Requires encoder checkpoints
      and considerably more GPU memory.
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_alignment.dataset import (
    SpineMRIEmbeddingDataset,
    alignment_collate_fn,
)
from model_alignment.models import (
    AttentionPooler,
    CLIPModel,
    LayerWiseSynthesizer,
    LinearProjectionHead,
    SynthesizerEncoder,
    TextProjectionHead,
    clip_contrastive_loss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding-level fusion weight (lightweight alternative to full synthesizer)
# ---------------------------------------------------------------------------

class EmbeddingRouter(nn.Module):
    """Learnable scalar that fuses pre-extracted T1/T2 embeddings."""

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit)

    def forward(self, t1_embs: torch.Tensor, t2_embs: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return a * t1_embs + (1 - a) * t2_embs


# ---------------------------------------------------------------------------
# Encoder helpers (for --mode image)
# ---------------------------------------------------------------------------

def build_dinov3_encoder(config_file: str, checkpoint_path: str):
    """Build a DINOv3 encoder and load pretrained weights."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pretraining"))

    from dinov3.configs.config import DinoV3SetupArgs, get_cfg_from_args
    from dinov3.models import build_model_from_cfg
    from dinov3.checkpointer import init_model_from_checkpoint_for_evals

    args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=checkpoint_path,
        output_dir="",
    )
    cfg = get_cfg_from_args(args)
    model, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    model.to_empty(device="cuda")
    init_model_from_checkpoint_for_evals(model, checkpoint_path, "teacher")
    model.eval()
    return model, embed_dim


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_embedding_mode(
    router,
    clip_model,
    text_encoder,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    writer,
):
    """Train embedding-level router with CLIP contrastive objective."""
    router.train()
    clip_model.train()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Router Epoch {epoch}")):
        t1_embs = batch["t1_embeddings"].to(device)
        t1_mask = batch["t1_mask"].to(device)
        t2_embs = batch["t2_embeddings"].to(device)
        t2_mask = batch["t2_mask"].to(device)
        reports = batch["reports"]

        encoding = tokenizer(
            reports, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            text_out = text_encoder(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        text_features = text_out.last_hidden_state[:, 0, :]

        fused_embs = router(t1_embs, t2_embs)
        fused_mask = t1_mask | t2_mask

        module = clip_model.module if isinstance(clip_model, DDP) else clip_model
        pooled = module.t1_pooler(fused_embs, fused_mask)
        image_repr = module.vision_proj.forward_t1(pooled)
        text_repr = module.text_proj(text_features)

        logit_scale = module.logit_scale.exp()
        logits_img = logit_scale * image_repr @ text_repr.t()
        logits_txt = logits_img.t()

        loss = clip_contrastive_loss(logits_img, logits_txt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(router.parameters()) + list(clip_model.parameters()), 1.0,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        writer.add_scalar("Loss/router_step", loss.item(), epoch * len(dataloader) + batch_idx)

    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("Loss/router_epoch", avg_loss, epoch)
    return avg_loss


def main(args):
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # ---- text encoder ----
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
    text_encoder = AutoModel.from_pretrained(args.text_encoder_name).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_embed_dim = text_encoder.config.hidden_size

    # ---- CLIP model (projections + poolers) ----
    vision_proj = LinearProjectionHead(in_features=args.vision_embed_dim, out_features=args.proj_dim)
    text_proj = TextProjectionHead(in_features=text_embed_dim, out_features=args.proj_dim)
    t1_pooler = AttentionPooler(input_dim=args.vision_embed_dim)
    t2_pooler = AttentionPooler(input_dim=args.vision_embed_dim)

    clip_model = CLIPModel(
        vision_proj=vision_proj, text_proj=text_proj,
        t1_pooler=t1_pooler, t2_pooler=t2_pooler,
        temperature=args.temperature,
    ).to(device)

    if args.clip_checkpoint:
        logger.info(f"Loading CLIP checkpoint from {args.clip_checkpoint}")
        ckpt = torch.load(args.clip_checkpoint, map_location=device)
        clip_model.load_state_dict(ckpt["clip_model_state"], strict=False)

    # ---- router ----
    if args.mode == "embedding":
        router = EmbeddingRouter().to(device)
        trainable_params = list(router.parameters()) + list(clip_model.parameters())
    else:
        logger.info("Building T1 and T2 encoders for layer-wise synthesis...")
        t1_enc, embed_dim = build_dinov3_encoder(args.t1_config, args.t1_checkpoint)
        t2_enc, _ = build_dinov3_encoder(args.t2_config, args.t2_checkpoint)
        num_layers = len(t1_enc.blocks)
        synthesizer = LayerWiseSynthesizer(num_layers=num_layers, embed_dim=embed_dim)
        router = SynthesizerEncoder(t1_enc, t2_enc, synthesizer).to(device)
        trainable_params = list(synthesizer.parameters()) + list(clip_model.parameters())

    if args.distributed:
        router = DDP(router, device_ids=[local_rank], find_unused_parameters=True)
        clip_model = DDP(clip_model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # ---- data ----
    dataset = SpineMRIEmbeddingDataset(
        data_json=args.data_json,
        t1_embedding_dir=args.t1_embedding_dir,
        t2_embedding_dir=args.t2_embedding_dir,
        report_csv=args.report_csv,
        max_slices=args.max_slices,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers,
        collate_fn=alignment_collate_fn, pin_memory=True, drop_last=True,
    )

    total_steps = args.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        avg_loss = train_embedding_mode(
            router, clip_model, text_encoder, tokenizer,
            dataloader, optimizer, scheduler, device, epoch, writer,
        )
        logger.info(f"Router Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if local_rank == 0:
            router_mod = router.module if isinstance(router, DDP) else router
            clip_mod = clip_model.module if isinstance(clip_model, DDP) else clip_model

            checkpoint = {
                "epoch": epoch,
                "clip_model_state": clip_mod.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "proj_t1_state": clip_mod.vision_proj.proj_t1.state_dict(),
                "proj_t2_state": clip_mod.vision_proj.proj_t2.state_dict(),
                "t1_mil_state": clip_mod.t1_pooler.state_dict(),
                "t2_mil_state": clip_mod.t2_pooler.state_dict(),
            }
            if args.mode == "embedding":
                checkpoint["router_state"] = router_mod.state_dict()
            else:
                checkpoint["synthesizer_state"] = router_mod.synthesizer.state_dict()

            torch.save(checkpoint, os.path.join(args.output_dir, f"router_epoch_{epoch}.pt"))
            torch.save(checkpoint, os.path.join(args.output_dir, "router_latest.pt"))

    writer.close()
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Router/Synthesizer training for SpineFM")
    parser.add_argument("--mode", type=str, default="embedding", choices=["embedding", "image"],
                        help="'embedding': learn fusion weight on pre-extracted features; "
                             "'image': full layer-wise synthesis (needs encoder checkpoints)")
    parser.add_argument("--t1_config", type=str, default=None)
    parser.add_argument("--t1_checkpoint", type=str, default=None)
    parser.add_argument("--t2_config", type=str, default=None)
    parser.add_argument("--t2_checkpoint", type=str, default=None)
    parser.add_argument("--clip_checkpoint", type=str, default=None)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--report_csv", type=str, required=True)
    parser.add_argument("--t1_embedding_dir", type=str, required=True)
    parser.add_argument("--t2_embedding_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/router_training")
    parser.add_argument("--text_encoder_name", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    parser.add_argument("--vision_embed_dim", type=int, default=1024)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_slices", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()
    main(args)
