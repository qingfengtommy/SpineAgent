import torch
import torch.nn as nn
from typing import Dict, List


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer applied on top of a frozen linear layer."""

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank

        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype

        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.randn(rank, original_layer.in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, rank, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_out + lora_out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = None,
    rank: int = 16,
    alpha: float = 16.0,
    dropout_p: float = 0.1,
) -> Dict[str, LoRALayer]:
    """Apply LoRA to targeted linear modules in a model."""
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj", "mlp.w12", "mlp.w3"]

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target_pattern in target_modules:
                if target_pattern in name:
                    lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout_p=dropout_p)
                    parent_names = name.split(".")[:-1]
                    parent = model
                    for parent_name in parent_names:
                        parent = getattr(parent, parent_name)
                    setattr(parent, name.split(".")[-1], lora_layer)
                    lora_layers[name] = lora_layer
                    break

    print(f"Applied LoRA to {len(lora_layers)} layers")
    return lora_layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in get_lora_parameters(model))
