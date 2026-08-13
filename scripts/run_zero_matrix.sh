#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:deepspeed}"
GPUS="${GPUS:-1 2 4 8}"
ZERO_STAGES="${ZERO_STAGES:-2 3}"
STEPS="${STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
REPEAT="${REPEAT:-3}"
OUT_DIR="${OUT_DIR:-results/zero_repeat3}"

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

run_ddp() {
  local nproc="$1"
  local output="${OUT_DIR}/ddp_${nproc}gpu.json"
  docker_run torchrun --standalone --nproc_per_node="${nproc}" -m minitrainbench train \
    --strategy ddp \
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

run_zero() {
  local stage="$1"
  local nproc="$2"
  local output="${OUT_DIR}/zero${stage}_${nproc}gpu.json"
  docker_run torchrun --standalone --nproc_per_node="${nproc}" -m minitrainbench deepspeed \
    --zero-stage "${stage}" \
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
  run_ddp "${nproc}"
done

for stage in ${ZERO_STAGES}; do
  for nproc in ${GPUS}; do
    run_zero "${stage}" "${nproc}"
  done
done

inputs=()
for nproc in ${GPUS}; do
  inputs+=("${OUT_DIR}/ddp_${nproc}gpu.json")
done
for stage in ${ZERO_STAGES}; do
  for nproc in ${GPUS}; do
    inputs+=("${OUT_DIR}/zero${stage}_${nproc}gpu.json")
  done
done

docker_run python3 -m minitrainbench report \
  --input "${inputs[@]}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'ZeRO 对比结果: %s\n' "${OUT_DIR}/report.md"
