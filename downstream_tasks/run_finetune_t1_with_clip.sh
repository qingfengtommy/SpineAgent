#!/usr/bin/env bash
# Condition prediction using the T1 encoder + CLIP-aligned projection (no router).
# Uses the attention pooler and projection head trained during CLIP alignment.
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)":"$(cd "$(dirname "$0")/../pretraining" && pwd)":${PYTHONPATH:-}

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_DIR="$(cd "${ROOT_DIR}/../pretraining" && pwd)"

CONFIG_FILE="${PRETRAIN_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
CHECKPOINT="${PRETRAIN_DIR}/dinov3/checkpoints/dinov3_vitl_pretrain_t1/teacher_checkpoint.pth"
CLIP_CKPT="${ROOT_DIR}/../model_alignment/checkpoints/clip_alignment/checkpoint_latest.pt"

# Data paths (modify according to your setup)
DATA_DIR="/path/to/condition_labels"
IMAGE_DIR="/path/to/mri_images/t1"
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
    echo "T1 encoder + CLIP proj | condition: ${CONDITION}"
    echo "=============================="

    export CUDA_VISIBLE_DEVICES=0
    python "${ROOT_DIR}/finetune.py" \
        --json_path "${DATA_DIR}/${CONDITION}_spinal_data_t1_png_with_labels.json" \
        --train_split_path "${SPLIT_DIR}/train_${CONDITION}_0.8_split.json" \
        --test_split_path "${SPLIT_DIR}/test_${CONDITION}_0.1_split.json" \
        --base_dir "${IMAGE_DIR}" \
        --save_dir "${ROOT_DIR}/checkpoints/${CONDITION}/dinov3_vitl_t1_clip" \
        --log_dir "${ROOT_DIR}/runs/${CONDITION}/dinov3_vitl_t1_clip" \
        --config_file "${CONFIG_FILE}" \
        --pretrained_weights "${CHECKPOINT}" \
        --output_dir "${ROOT_DIR}/output/${CONDITION}/t1_clip" \
        --feature_dim 1024 \
        --normalize \
        --num_epochs 3 \
        --batch_size 128 \
        --learning_rate 5e-4 \
        --pool MIL \
        --cache_home "${ROOT_DIR}/cache_embeddings" \
        --project_ckpt "${CLIP_CKPT}" \
        --projection_type patient_attn
done
