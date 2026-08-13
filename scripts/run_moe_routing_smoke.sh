#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-nccl}"
OUT_DIR="${OUT_DIR:-results/moe_routing}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:-1.25}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

output="${OUT_DIR}/moe_routing_${NPROC}gpu.json"

if [[ "${NPROC}" == "1" ]]; then
  docker_run python3 -m minitrainbench moe route \
    --device "${DEVICE}" \
    --backend "${BACKEND}" \
    --tokens-per-rank "${TOKENS_PER_RANK}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --num-experts "${NUM_EXPERTS}" \
    --capacity-factor "${CAPACITY_FACTOR}" \
    --output "/workspace/${output}"
else
  docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench moe route \
    --device "${DEVICE}" \
    --backend "${BACKEND}" \
    --tokens-per-rank "${TOKENS_PER_RANK}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --num-experts "${NUM_EXPERTS}" \
    --capacity-factor "${CAPACITY_FACTOR}" \
    --output "/workspace/${output}"
fi

docker_run python3 -m minitrainbench report \
  --input "${output}" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'MoE routing smoke 结果: %s\n' "${OUT_DIR}/report.md"
