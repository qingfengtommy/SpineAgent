#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_DIR="$(cd "${ROOT_DIR}/../pretraining" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/router_training"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/router_training.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Paths (modify according to your setup)
T1_CONFIG="${PRETRAIN_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
T1_CHECKPOINT="${PRETRAIN_DIR}/dinov3/checkpoints/dinov3_vitl_pretrain_t1/teacher_checkpoint.pth"
T2_CONFIG="${PRETRAIN_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
T2_CHECKPOINT="${PRETRAIN_DIR}/dinov3/checkpoints/dinov3_vitl_pretrain_t2/teacher_checkpoint.pth"
CLIP_CHECKPOINT="${ROOT_DIR}/checkpoints/clip_alignment/checkpoint_latest.pt"

DATA_JSON="/path/to/pretraining_data.json"
REPORT_CSV="/path/to/clinical_reports.csv"
T1_EMBEDDING_DIR="/path/to/t1_embeddings"
T2_EMBEDDING_DIR="/path/to/t2_embeddings"

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29511 \
  "${ROOT_DIR}/train_router.py" \
  --t1_config "${T1_CONFIG}" \
  --t1_checkpoint "${T1_CHECKPOINT}" \
  --t2_config "${T2_CONFIG}" \
  --t2_checkpoint "${T2_CHECKPOINT}" \
  --clip_checkpoint "${CLIP_CHECKPOINT}" \
  --data_json "${DATA_JSON}" \
  --report_csv "${REPORT_CSV}" \
  --t1_embedding_dir "${T1_EMBEDDING_DIR}" \
  --t2_embedding_dir "${T2_EMBEDDING_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --proj_dim 1024 \
  --epochs 5 \
  --batch_size 80 \
  --learning_rate 4e-5 \
  --temperature 0.07 \
  --max_slices 64 \
  --num_workers 8 \
  --distributed \
  > "${LOG_FILE}" 2>&1 &

echo "Started router training. Logs: ${LOG_FILE}"
