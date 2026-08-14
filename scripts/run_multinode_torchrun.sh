#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:?请设置 NODE_RANK，例如 0 或 1}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:?请设置 RDZV_ENDPOINT，例如 10.0.0.1:29500}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
OUT_DIR="${OUT_DIR:-results/multinode}"
COMMAND="${COMMAND:-python3 -m minitrainbench doctor --expected-world-size $((NNODES * NPROC_PER_NODE))}"

mkdir -p "${OUT_DIR}"

MINITRAINBENCH_DOCKER_ARGS=(
  -e NCCL_DEBUG \
  -e NCCL_SOCKET_IFNAME \
  -e NCCL_IB_DISABLE \
  -e NCCL_IB_HCA \
  -e NCCL_ASYNC_ERROR_HANDLING \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING
)
minitrainbench_docker_run "${IMAGE}" \
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend="${RDZV_BACKEND}" \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    ${COMMAND}
