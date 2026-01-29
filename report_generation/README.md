## SpineAgent Report Generation

This directory contains the SpineAgent adaptation of LLaVA for **spine MRI report generation and related tasks** (impression summarization and generation).

- **Core model code**: `llava/model/`
  - `builder.py`: utilities to load base LLaMA, attach dual DINOv3 vision towers, and construct the multimodal model.
  - `llava_arch.py`: main multimodal architecture, vision tower + projector wiring, and special token handling.
  - `multimodal_encoder/dual_dinov3_encoder.py`: dual‑modality DINOv3 vision backbone (T1/T2).
  - `multimodal_projector/`: CoCa‑style attentional pooling and vision→language projector.
- **Training entrypoints**: `llava/train/`
  - `train.py`: argument definitions, dataset construction, and `Trainer` setup.
  - `train_mem.py`: thin wrapper that configures `sys.path` and calls `train()` (used by the shell scripts).
- **Scripts**: `llava/scripts/`
  - `spinal_agent_pretrain.sh`: stage‑1 pretraining of the multimodal projector and dual DINOv3 encoder on spine MRI + reports.
  - `spinal_agent_finetune.sh`: stage‑2 supervised finetuning that reuses the pre‑trained projector via `pretrain_mm_mlp_adapter`.
  - You can set up hyper-parameters based on your project and write your own dataset class

### Running (internal paths vs. open source)

The current scripts reference **internal, absolute paths** for datasets and checkpoints (e.g. hospital MRI storage locations).  
Before open‑sourcing, you should:

1. **Remove or anonymize all absolute paths** in the shell scripts and Python files.
2. Replace them with example placeholders (e.g. `/path/to/images.json`, `/path/to/dinov3_checkpoint`) and document the expected formats.


