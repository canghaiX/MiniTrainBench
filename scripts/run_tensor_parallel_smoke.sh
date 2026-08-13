#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-nccl}"
OUT_DIR="${OUT_DIR:-results/tensor_parallel}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-8}"
IN_FEATURES="${IN_FEATURES:-1024}"
OUT_FEATURES="${OUT_FEATURES:-4096}"
MLP_HIDDEN_FEATURES="${MLP_HIDDEN_FEATURES:-${OUT_FEATURES}}"
MLP_OUT_FEATURES="${MLP_OUT_FEATURES:-${IN_FEATURES}}"
SP_HIDDEN_SIZE="${SP_HIDDEN_SIZE:-${IN_FEATURES}}"
SP_DROPOUT="${SP_DROPOUT:-0.1}"
ATOL="${ATOL:-1e-3}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

basic_output="${OUT_DIR}/tp_check_${NPROC}gpu.json"
mlp_output="${OUT_DIR}/tp_mlp_${NPROC}gpu.json"
sequence_output="${OUT_DIR}/sequence_parallel_${NPROC}gpu.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench tp check \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --in-features "${IN_FEATURES}" \
  --out-features "${OUT_FEATURES}" \
  --atol "${ATOL}" \
  --output "/workspace/${basic_output}"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench tp mlp \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --in-features "${IN_FEATURES}" \
  --hidden-features "${MLP_HIDDEN_FEATURES}" \
  --out-features "${MLP_OUT_FEATURES}" \
  --atol "${ATOL}" \
  --output "/workspace/${mlp_output}"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench tp sequence \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --hidden-size "${SP_HIDDEN_SIZE}" \
  --dropout "${SP_DROPOUT}" \
  --atol "${ATOL}" \
  --output "/workspace/${sequence_output}"

docker_run python3 -m minitrainbench report \
  --input "${basic_output}" "${mlp_output}" "${sequence_output}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'Tensor Parallel 正确性结果: %s\n' "${OUT_DIR}/report.md"
