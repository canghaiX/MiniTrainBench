#!/usr/bin/env bash
set -euo pipefail

: "${MEGATRON_DIR:?请设置 MEGATRON_DIR=/path/to/Megatron-LM}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.18.2}"
MEGATRON_IMAGE="${MEGATRON_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
OUT_DIR="${OUT_DIR:-results/megatron_smoke}"
TP="${TP:-1}"
PP="${PP:-1}"
WORLD_SIZE="${WORLD_SIZE:-8}"
NAME="${NAME:-tp${TP}_pp${PP}}"
MEASURED_ITERS="${MEASURED_ITERS:-5}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_HEADS="${NUM_HEADS:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"

if [[ "${OUT_DIR}" == /* || "/${OUT_DIR}/" == *"/../"* ]]; then
  echo "OUT_DIR 必须是当前仓库下的相对路径" >&2
  exit 2
fi

if [[ ! -f "${MEGATRON_DIR}/pretrain_gpt.py" ]]; then
  echo "MEGATRON_DIR 中未找到 pretrain_gpt.py: ${MEGATRON_DIR}" >&2
  exit 2
fi
if (( WORLD_SIZE % (TP * PP) != 0 )); then
  echo "WORLD_SIZE 必须能被 TP*PP 整除" >&2
  exit 2
fi
DP=$((WORLD_SIZE / TP / PP))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
  echo "GLOBAL_BATCH_SIZE 必须能被 MICRO_BATCH_SIZE*DP 整除" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/records" "${OUT_DIR}/tensorboard"
MEGATRON_COMMIT="$(git -C "${MEGATRON_DIR}" rev-parse HEAD)"
if ! git -C "${MEGATRON_DIR}" rev-parse --verify "${MEGATRON_REF}^{commit}" >/dev/null 2>&1; then
  echo "外部 Megatron 仓库中不存在固定 ref: ${MEGATRON_REF}" >&2
  exit 2
fi
EXPECTED_COMMIT="$(git -C "${MEGATRON_DIR}" rev-parse "${MEGATRON_REF}^{commit}")"
if [[ "${MEGATRON_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Megatron HEAD 与 ${MEGATRON_REF} 不一致；脚本不会修改外部仓库" >&2
  exit 2
fi

TRAIN_ITERS=$((WARMUP_ITERS + MEASURED_ITERS))
LOG_PATH="${OUT_DIR}/logs/${NAME}.log"
RECORD_PATH="${OUT_DIR}/records/${NAME}.json"
COMMAND_PATH="${OUT_DIR}/logs/${NAME}.command.txt"
ENVIRONMENT_PATH="${OUT_DIR}/environment.json"
ENVIRONMENT_LOG_PATH="${OUT_DIR}/logs/environment_probe.log"

config_json="$(printf '{\"name\":\"%s\",\"tp\":%s,\"pp\":%s,\"dp\":%s,\"world_size\":%s,\"micro_batch_size\":%s,\"global_batch_size\":%s,\"seq_length\":%s,\"measured_iters\":%s,\"warmup_iters\":%s,\"megatron_ref\":\"%s\",\"megatron_commit\":\"%s\",\"image\":\"%s\",\"transformer_impl\":\"%s\",\"environment_file\":\"%s\",\"command_file\":\"%s\"}' "${NAME}" "${TP}" "${PP}" "${DP}" "${WORLD_SIZE}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${SEQ_LENGTH}" "${MEASURED_ITERS}" "${WARMUP_ITERS}" "${MEGATRON_REF}" "${MEGATRON_COMMIT}" "${MEGATRON_IMAGE}" "${TRANSFORMER_IMPL}" "${ENVIRONMENT_PATH}" "${COMMAND_PATH}")"

if [[ ! -f "${ENVIRONMENT_PATH}" ]]; then
  environment_probe="$(cat <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

import torch

try:
    nccl_version = torch.cuda.nccl.version()
except Exception:
    nccl_version = None
try:
    megatron_core_version = importlib.metadata.version("megatron-core")
except importlib.metadata.PackageNotFoundError:
    megatron_core_version = os.environ["MEGATRON_REF"]

payload = {
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "nccl_version": nccl_version,
    "megatron_core_version": megatron_core_version,
    "megatron_ref": os.environ["MEGATRON_REF"],
    "megatron_commit": os.environ["MEGATRON_COMMIT"],
    "image": os.environ["MEGATRON_IMAGE"],
}
Path(os.environ["ENVIRONMENT_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
)"
  set +e
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -v "${MEGATRON_DIR}:/megatron:ro" \
    -e PYTHONPATH=/megatron -e MEGATRON_REF="${MEGATRON_REF}" \
    -e MEGATRON_COMMIT="${MEGATRON_COMMIT}" -e MEGATRON_IMAGE="${MEGATRON_IMAGE}" \
    -e ENVIRONMENT_PATH="/workspace/${ENVIRONMENT_PATH}" \
    -w /megatron "${MEGATRON_IMAGE}" python3 -c "${environment_probe}" \
    >"${ENVIRONMENT_LOG_PATH}" 2>&1
  environment_returncode=$?
  set -e
  if (( environment_returncode != 0 )); then
    PYTHONPATH=src python3 -m minitrainbench.evidence megatron-record \
      --config-json "${config_json}" --log "${ENVIRONMENT_LOG_PATH}" \
      --returncode "${environment_returncode}" --output "${RECORD_PATH}"
    echo "Megatron 容器环境探针失败: ${ENVIRONMENT_LOG_PATH}" >&2
    exit "${environment_returncode}"
  fi
fi

megatron_args=(
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" /megatron/pretrain_gpt.py
  --mock-data --tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE}"
  --num-layers "${NUM_LAYERS}" --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "$((4 * HIDDEN_SIZE))" --num-attention-heads "${NUM_HEADS}"
  --seq-length "${SEQ_LENGTH}" --max-position-embeddings "${SEQ_LENGTH}"
  --micro-batch-size "${MICRO_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}" --lr-decay-iters "${TRAIN_ITERS}"
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 1
  --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95
  --bf16 --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --normalization RMSNorm --swiglu --disable-bias-linear
  --untie-embeddings-and-output-weights --position-embedding-type rope
  --attention-dropout 0.0 --hidden-dropout 0.0
  --transformer-impl "${TRANSFORMER_IMPL}" --distributed-backend nccl
  --eval-iters 0 --eval-interval 100000 --log-interval 1 --log-throughput
  --tensorboard-dir "/workspace/${OUT_DIR}/tensorboard/${NAME}"
  --log-timers-to-tensorboard --log-memory-to-tensorboard
)
if (( TP > 1 )); then
  megatron_args+=(--sequence-parallel)
fi

command=(
  docker run --rm --gpus all --ipc=host --network=host
  -v "${PWD}:/workspace" -v "${MEGATRON_DIR}:/megatron:ro"
  -e PYTHONPATH=/megatron
  -w /megatron "${MEGATRON_IMAGE}" "${megatron_args[@]}"
)
printf '%q ' "${command[@]}" > "${COMMAND_PATH}"
printf '\n' >> "${COMMAND_PATH}"

set +e
timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" "${command[@]}" \
  >"${LOG_PATH}" 2>&1
returncode=$?
set -e

PYTHONPATH=src python3 -m minitrainbench.evidence megatron-record \
  --config-json "${config_json}" --log "${LOG_PATH}" --returncode "${returncode}" \
  --output "${RECORD_PATH}"

printf '%s\n' "${RECORD_PATH}"
