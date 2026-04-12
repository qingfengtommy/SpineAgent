#!/usr/bin/env bash
# Condition prediction using the router/synthesizer (fused T1+T2 embeddings).
# This mode loads both T1 and T2 encoders, runs each image through the
# trained layer-wise synthesizer to produce series-agnostic embeddings,
# then trains a linear classifier on the fused features.
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)":"$(cd "$(dirname "$0")/../pretraining" && pwd)":${PYTHONPATH:-}

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_DIR="$(cd "${ROOT_DIR}/../pretraining" && pwd)"
ALIGN_DIR="$(cd "${ROOT_DIR}/../model_alignment" && pwd)"

T1_CONFIG="${PRETRAIN_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
T1_CHECKPOINT="${PRETRAIN_DIR}/dinov3/checkpoints/dinov3_vitl_pretrain_t1/teacher_checkpoint.pth"
T2_CONFIG="${PRETRAIN_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
T2_CHECKPOINT="${PRETRAIN_DIR}/dinov3/checkpoints/dinov3_vitl_pretrain_t2/teacher_checkpoint.pth"
ROUTER_CKPT="${ALIGN_DIR}/checkpoints/router_training/router_latest.pt"
CLIP_CKPT="${ALIGN_DIR}/checkpoints/clip_alignment/checkpoint_latest.pt"

# Data paths (modify according to your setup)
DATA_DIR="/path/to/condition_labels"
IMAGE_DIR="/path/to/mri_images"
SPLIT_DIR="${ROOT_DIR}/splits"
mkdir -p "${SPLIT_DIR}"

CONDITIONS=(
    "canal"
    "par"
    "compression"
    "edema"
    "inflammation"
    "neuroforaminal"
    "spondylolisthesis"
    "cyst"
    "herniation"
    "hemangioma"
    "lipomatosis"
    "lesion"
    "fracture"
    "disc_height"
    "subarticular"
    "synovial"
)

for CONDITION in "${CONDITIONS[@]}"; do
    echo "=============================="
    echo "Router (fused T1+T2) | condition: ${CONDITION}"
    echo "=============================="

    export CUDA_VISIBLE_DEVICES=0
    python "${ROOT_DIR}/finetune.py" \
        --json_path "${DATA_DIR}/${CONDITION}_spinal_data_t1_t2_png_with_labels.json" \
        --train_split_path "${SPLIT_DIR}/train_${CONDITION}_all_0.8_split.json" \
        --test_split_path "${SPLIT_DIR}/test_${CONDITION}_all_0.1_split.json" \
        --base_dir "${IMAGE_DIR}" \
        --save_dir "${ROOT_DIR}/checkpoints/${CONDITION}/dinov3_vitl_router" \
        --log_dir "${ROOT_DIR}/runs/${CONDITION}/dinov3_vitl_router" \
        --output_dir "${ROOT_DIR}/output/${CONDITION}/router" \
        --feature_dim 1024 \
        --normalize \
        --num_epochs 3 \
        --batch_size 128 \
        --learning_rate 5e-4 \
        --pool MIL \
        --cache_home "${ROOT_DIR}/cache_embeddings" \
        --use_router \
        --t1_config "${T1_CONFIG}" \
        --t1_checkpoint "${T1_CHECKPOINT}" \
        --t2_config "${T2_CONFIG}" \
        --t2_checkpoint "${T2_CHECKPOINT}" \
        --router_checkpoint "${ROUTER_CKPT}" \
        --project_ckpt "${CLIP_CKPT}" \
        --projection_type patient_attn
done
