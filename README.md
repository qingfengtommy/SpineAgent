# SpineAgent

A multi-agent system for spine MRI report generation from multi-sequence imaging.

SpineAgent is built upon **SpineFM**, a multi-sequence MRI foundation model trained on routine clinical data from 32,047 patients and 453,683 MRI series, comprising a total of 13,441,191 MRI slices. The system decomposes report generation into clinically grounded subtasks handled by 37 specialized agents for diagnosis, pathological-region localization, and clinically-similar-cases retrieval, whose outputs are integrated by a Medical Report Agent for end-to-end report generation.

## Highlights

- Multi-sequence MRI foundation model with separate T1/T2 DINOv3-based encoders and a continual training strategy via a synthesizer module for arbitrary MRI sequences.
- CLIP-based tri-modal alignment (T1 images, T2 images, clinical reports) using BiomedBERT as the text encoder.
- State-of-the-art performance on 17 spinal condition prediction tasks with mean 10.8% AUROC improvement.
- Strong cross-manufacturer and cross-cohort generalization.
- Pathology localization via two-phase segmentation agents (slice selection + region segmentation).
- Cross-modal image-report retrieval with 56.38% improvement in Recall@5.
- End-to-end report generation with structured agent outputs as tokens in a LLaVA-based Medical Report Agent.

## Repository Structure

```
SpineAgent/
├── pretraining/            # DINOv3-based encoder pretraining
│   ├── dinov3/             # DINOv3 framework (models, losses, data, training)
│   ├── run_dinov3_pretrain_t1.sh
│   ├── run_dinov3_pretrain_t2.sh
│   ├── run_dinov3_spinal_gram_t1.sh
│   └── run_dinov3_spinal_gram_t2.sh
├── model_alignment/        # Text-image alignment and router training
│   ├── clip/               # OpenAI CLIP (cloned)
│   ├── models.py           # Synthesizer, projection heads, CLIP model
│   ├── dataset.py          # Alignment dataset utilities
│   ├── train_clip.py       # CLIP contrastive alignment training
│   ├── train_router.py     # Router/synthesizer training
│   ├── run_clip_alignment.sh
│   └── run_router_training.sh
├── downstream_tasks/       # Condition prediction with linear probing
│   ├── finetune.py         # Main training script (supports direct encoder & router)
│   ├── utils.py            # Dataset, evaluation, MIL classifiers
│   ├── lora.py             # LoRA implementation
│   ├── run_finetune_t1.sh             # T1 encoder, no router
│   ├── run_finetune_t2.sh             # T2 encoder, no router
│   ├── run_finetune_t1_with_clip.sh   # T1 encoder + CLIP projection
│   └── run_finetune_router.sh         # Router (fused T1+T2 via synthesizer)
├── report_generation/      # LLaVA-based Medical Report Agent
│   └── llava/              # Adapted LLaVA with dual DINOv3 vision encoders
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n spineagent python=3.10
conda activate spineagent
pip install -r requirements.txt
```

## Training Pipeline

### 1. Pre-training (DINOv3 Encoders)

Train separate T1 and T2 encoders using the DINOv3 self-supervised framework:

```bash
cd pretraining
bash run_dinov3_pretrain_t1.sh   # Phase 1: T1 encoder
bash run_dinov3_pretrain_t2.sh   # Phase 1: T2 encoder
bash run_dinov3_spinal_gram_t1.sh  # Phase 2: Gram-regularized anchor training
bash run_dinov3_spinal_gram_t2.sh
```

### 2. Text-Image Alignment (CLIP)

Align T1/T2 embeddings with clinical report embeddings from BiomedBERT:

```bash
cd model_alignment
bash run_clip_alignment.sh
```

### 3. Router/Synthesizer Training

Train the layer-wise synthesizer for fusing T1/T2 representations on non-T1/T2 sequences:

```bash
cd model_alignment
bash run_router_training.sh
```

### 4. Downstream Condition Prediction

Run linear probing on frozen SpineFM embeddings for 17 spinal conditions:

```bash
cd downstream_tasks

# Without router (direct single-encoder evaluation)
bash run_finetune_t1.sh             # T1 encoder only
bash run_finetune_t2.sh             # T2 encoder only
bash run_finetune_t1_with_clip.sh   # T1 encoder + CLIP-aligned projection

# With router (fused T1+T2 via trained synthesizer)
bash run_finetune_router.sh         # All series through synthesizer
```

### 5. Report Generation

Train the LLaVA-based Medical Report Agent:

```bash
cd report_generation/llava/scripts
bash spinal_agent_pretrain.sh   # Stage 1: Projector warm-up
bash spinal_agent_finetune.sh   # Stage 2: Language decoder adaptation
```

## Data

This project is trained on a large, de-identified clinical spine MRI cohort from the University of Washington Medical Center comprising 32,047 patients, 453,683 MRI series, and 13,441,191 slices. Due to patient privacy considerations and institutional regulations, the imaging data is not publicly available. For academic collaboration or data access inquiries, please contact the corresponding authors (nmcross@uw.edu, swang@cs.washington.edu).

For external evaluation, we use the publicly available [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbarspine-degenerative-classification) dataset.


## Citation

```bibtex
@article{
  title={A multi-agent system for spine MRI report generation from multi-sequence imaging},
  author={Xiao, Zhiping and Yang, Junwei and Sun, Gongbo and Zhang, Han and Xu, Hanwen and Yao, Yi and Miller, Zachary D. and King III, William E. and Kanani, Mohammed M. and Andre, Jalal B. and Chu, Sammy and Kinahan, Paul E. and Cross, Nathan M. and Wang, Sheng},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.
