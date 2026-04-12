#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/clip_alignment"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/clip_alignment.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Paths (modify according to your setup)
DATA_JSON="/path/to/pretraining_data.json"
REPORT_CSV="/path/to/clinical_reports.csv"
T1_EMBEDDING_DIR="/path/to/t1_embeddings"
T2_EMBEDDING_DIR="/path/to/t2_embeddings"

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29510 \
  "${ROOT_DIR}/train_clip.py" \
  --data_json "${DATA_JSON}" \
  --report_csv "${REPORT_CSV}" \
  --t1_embedding_dir "${T1_EMBEDDING_DIR}" \
  --t2_embedding_dir "${T2_EMBEDDING_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --vision_embed_dim 1024 \
  --proj_dim 1024 \
  --epochs 5 \
  --batch_size 80 \
  --learning_rate 4e-5 \
  --temperature 0.07 \
  --max_slices 64 \
  --num_workers 8 \
  --distributed \
  > "${LOG_FILE}" 2>&1 &

echo "Started CLIP alignment training. Logs: ${LOG_FILE}"
