#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
GPUS="${GPUS:-2 4 8}"
OUT_DIR="${OUT_DIR:-results/moe_comm}"
SIZES="${SIZES:-1024,1048576,16777216}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
ALL_TO_ALL_MODE="${ALL_TO_ALL_MODE:-both}"

mkdir -p "${OUT_DIR}"

docker_run() {
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" "$@"
}

inputs=()
for nproc in ${GPUS}; do
  output="${OUT_DIR}/all_to_all_${nproc}gpu.json"
  docker_run torchrun --standalone --nproc_per_node="${nproc}" -m minitrainbench comm \
    --device cuda \
    --backend nccl \
    --operations all_to_all \
    --all-to-all-mode "${ALL_TO_ALL_MODE}" \
    --sizes "${SIZES}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "/workspace/${output}"
  inputs+=("${output}")
done

docker_run python3 -m minitrainbench report \
  --input "${inputs[@]}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'MoE all-to-all 通信结果: %s\n' "${OUT_DIR}/report.md"
