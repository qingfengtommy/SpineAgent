#!/usr/bin/env bash
set -euo pipefail
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
export PYTHONPATH="/dinov3:${PYTHONPATH:-}"

ROOT_DIR="/home/gongbsun/SpineAgent/pretraining/"

# Paths
TRAIN_PY="dinov3/train/train.py"
CONFIG_YAML="${ROOT_DIR}/dinov3/configs/train/dinov3_vitl16_pretrain.yaml"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/dinov3_vitl_pretrain_t2"
LOG_DIR="${ROOT_DIR}/spinal_log"
LOG_FILE="${LOG_DIR}/dinov3_vitl_pretrain_t2.log"

# Ensure required directories exist
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Launch training
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port 29501 \
  "${TRAIN_PY}" \
  --config-file "${CONFIG_YAML}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality "t2" \
  > "${LOG_FILE}" 2>&1 &

echo "Started training. Logs: ${LOG_FILE}"

