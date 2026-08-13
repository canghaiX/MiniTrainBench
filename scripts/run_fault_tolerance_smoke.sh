#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
DEVICE="${DEVICE:-cpu}"
BACKEND="${BACKEND:-gloo}"
STRATEGY="${STRATEGY:-ddp}"
PRECISION="${PRECISION:-fp32}"
OUT_DIR="${OUT_DIR:-results/fault_tolerance}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-8}"
VOCAB_SIZE="${VOCAB_SIZE:-64}"
D_MODEL="${D_MODEL:-16}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.2}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

summary="${OUT_DIR}/fault_tolerance.json"

docker_run python3 -m minitrainbench fault smoke \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --strategy "${STRATEGY}" \
  --precision "${PRECISION}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --vocab-size "${VOCAB_SIZE}" \
  --d-model "${D_MODEL}" \
  --n-heads "${N_HEADS}" \
  --n-layers "${N_LAYERS}" \
  --dropout "${DROPOUT}" \
  --checkpoint-dir "/workspace/${OUT_DIR}/continuous" \
  --output "/workspace/${summary}"

docker_run python3 -m minitrainbench report \
  --input "${summary}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'Fault tolerance smoke 结果: %s\n' "${OUT_DIR}/report.md"
