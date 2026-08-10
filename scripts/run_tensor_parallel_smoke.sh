#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-nccl}"
OUT_DIR="${OUT_DIR:-results/tensor_parallel}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-8}"
IN_FEATURES="${IN_FEATURES:-1024}"
OUT_FEATURES="${OUT_FEATURES:-4096}"
ATOL="${ATOL:-1e-4}"

mkdir -p "${OUT_DIR}"

docker_run() {
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" "$@"
}

output="${OUT_DIR}/tp_check_${NPROC}gpu.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench tp check \
  --device "${DEVICE}" \
  --backend "${BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --in-features "${IN_FEATURES}" \
  --out-features "${OUT_FEATURES}" \
  --atol "${ATOL}" \
  --output "/workspace/${output}"

docker_run python3 -m minitrainbench report \
  --input "${output}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'Tensor Parallel 正确性结果: %s\n' "${OUT_DIR}/report.md"
