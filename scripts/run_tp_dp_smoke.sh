#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-4}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-nccl}"
OUT_DIR="${OUT_DIR:-results/tp_dp}"
TP_SIZE="${TP_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-8}"
IN_FEATURES="${IN_FEATURES:-1024}"
HIDDEN_FEATURES="${HIDDEN_FEATURES:-4096}"
OUT_FEATURES="${OUT_FEATURES:-1024}"
STEPS="${STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
PRECISION="${PRECISION:-bf16}"

if (( NPROC < 1 || TP_SIZE < 1 || NPROC % TP_SIZE != 0 )); then
  printf 'NPROC 必须是正数，TP_SIZE 必须是 NPROC 的正整数因子\n' >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

output="${OUT_DIR}/tp${TP_SIZE}_dp$((NPROC / TP_SIZE)).json"

minitrainbench_docker_run "${IMAGE}" \
  torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench tp train \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --tp-size "${TP_SIZE}" \
  --precision "${PRECISION}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --in-features "${IN_FEATURES}" \
  --hidden-features "${HIDDEN_FEATURES}" \
  --out-features "${OUT_FEATURES}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --output "/workspace/${output}"

minitrainbench_docker_run "${IMAGE}" \
  python3 -m minitrainbench report \
  --input "${output}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'TP + DP 训练结果: %s\n' "${OUT_DIR}/report.md"
