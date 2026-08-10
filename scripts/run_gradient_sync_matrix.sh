#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
OUT_DIR="${OUT_DIR:-results/gradient_sync}"
STEPS="${STEPS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
REPEAT="${REPEAT:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"

mkdir -p "${OUT_DIR}"

docker_run() {
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" "$@"
}

run_train() {
  local strategy="$1"
  local mode="$2"
  local output="${OUT_DIR}/${strategy}_${mode}_${NPROC}gpu.json"
  docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
    --device "${DEVICE}" \
    --strategy "${strategy}" \
    --precision "${PRECISION}" \
    --gradient-sync-mode "${mode}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
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

run_train ddp auto
run_train ddp every
run_train fsdp auto
run_train fsdp last

docker_run python3 -m minitrainbench report \
  --input "${OUT_DIR}/ddp_auto_${NPROC}gpu.json" \
          "${OUT_DIR}/ddp_every_${NPROC}gpu.json" \
          "${OUT_DIR}/fsdp_auto_${NPROC}gpu.json" \
          "${OUT_DIR}/fsdp_last_${NPROC}gpu.json" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'gradient sync 对比结果: %s\n' "${OUT_DIR}/report.md"
