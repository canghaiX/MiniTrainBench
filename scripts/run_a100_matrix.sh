#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
GPUS="${GPUS:-1 2 4 8}"
COMM_NPROC="${COMM_NPROC:-8}"
STEPS="${STEPS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
REPEAT="${REPEAT:-1}"
OUT_DIR="${OUT_DIR:-results}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

run_train() {
  local strategy="$1"
  local nproc="$2"
  local output="${OUT_DIR}/${strategy}_${nproc}gpu.json"
  docker_run torchrun --standalone --nproc_per_node="${nproc}" -m minitrainbench train \
    --strategy "${strategy}" \
    --precision bf16 \
    --batch-size "${BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --vocab-size "${VOCAB_SIZE}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --repeat "${REPEAT}" \
    --output "/workspace/${output}"
}

for nproc in ${GPUS}; do
  run_train ddp "${nproc}"
done

for nproc in ${GPUS}; do
  run_train fsdp "${nproc}"
done

docker_run torchrun --standalone --nproc_per_node="${COMM_NPROC}" -m minitrainbench comm \
  --device cuda \
  --backend nccl \
  --sizes 1024,1048576,16777216 \
  --warmup 10 \
  --iters 50 \
  --output "/workspace/${OUT_DIR}/nccl_${COMM_NPROC}gpu.json"

inputs=()
for nproc in ${GPUS}; do
  inputs+=("${OUT_DIR}/ddp_${nproc}gpu.json")
done
for nproc in ${GPUS}; do
  inputs+=("${OUT_DIR}/fsdp_${nproc}gpu.json")
done
inputs+=("${OUT_DIR}/nccl_${COMM_NPROC}gpu.json")

docker_run python3 -m minitrainbench report \
  --input "${inputs[@]}" \
  --output "/workspace/${OUT_DIR}/report.md"
