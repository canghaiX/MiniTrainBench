#!/usr/bin/env bash
set -uo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
DEEPSPEED_IMAGE="${DEEPSPEED_IMAGE:-minitrainbench:deepspeed}"
OUT_DIR="${OUT_DIR:-results/memory_pressure}"
WORLD_SIZE="${WORLD_SIZE:-8}"
STRATEGIES="${STRATEGIES:-ddp fsdp zero2 zero3}"
TIERS="${TIERS:-small medium large stress}"
STEPS="${STEPS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
REPEAT="${REPEAT:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

if [[ "${OUT_DIR}" == /* || "/${OUT_DIR}/" == *"/../"* ]]; then
  echo "OUT_DIR 必须是当前仓库下的相对路径" >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPUS}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy_gpus="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$2 + 0 > 1024 {gsub(/ /, "", $1); print $1}' \
      | paste -sd, -
  )"
  if [[ -n "${busy_gpus}" ]]; then
    echo "GPU ${busy_gpus} 已有超过 1024 MB 显存占用；拒绝污染性能证据。" >&2
    echo "确认允许并发采样时设置 ALLOW_BUSY_GPUS=1。" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/records" "${OUT_DIR}/raw"

tier_config() {
  case "$1" in
    small)  echo "8192 512 8 6 256 2" ;;
    medium) echo "16384 1024 16 12 512 1" ;;
    large)  echo "32768 1536 24 24 512 1" ;;
    stress) echo "32768 2560 32 32 512 1" ;;
    *) echo "未知模型档位: $1" >&2; return 2 ;;
  esac
}

run_case() {
  local tier="$1"
  local strategy="$2"
  local vocab_size d_model n_heads n_layers seq_length batch_size
  read -r vocab_size d_model n_heads n_layers seq_length batch_size < <(tier_config "${tier}")

  local benchmark_id="${tier}_${strategy}_${WORLD_SIZE}gpu"
  local result_path="${OUT_DIR}/raw/${benchmark_id}.json"
  local record_path="${OUT_DIR}/records/${benchmark_id}.json"
  local log_path="${OUT_DIR}/logs/${benchmark_id}.log"
  local container_result_path="/workspace/${result_path}"
  local image="${IMAGE}"
  local -a benchmark_args

  if [[ "${strategy}" == zero* ]]; then
    image="${DEEPSPEED_IMAGE}"
    benchmark_args=(
      torchrun --standalone --nproc_per_node="${WORLD_SIZE}"
      -m minitrainbench deepspeed --zero-stage "${strategy#zero}"
    )
  else
    benchmark_args=(
      torchrun --standalone --nproc_per_node="${WORLD_SIZE}"
      -m minitrainbench train --strategy "${strategy}"
    )
  fi

  benchmark_args+=(
    --precision bf16 --batch-size "${batch_size}" --seq-length "${seq_length}"
    --vocab-size "${vocab_size}" --d-model "${d_model}" --n-heads "${n_heads}"
    --n-layers "${n_layers}" --steps "${STEPS}" --warmup-steps "${WARMUP_STEPS}"
    --repeat "${REPEAT}" --output "${container_result_path}"
  )
  if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
    benchmark_args+=(--activation-checkpointing)
  fi

  local -a command=(
    docker run --rm --gpus all --ipc=host --network=host
    -v "${PWD}:/workspace" -w /workspace "${image}" "${benchmark_args[@]}"
  )
  printf '运行 %s\n' "${benchmark_id}"
  rm -f "${result_path}" "${record_path}"
  printf '%q ' "${command[@]}" > "${OUT_DIR}/logs/${benchmark_id}.command.txt"
  printf '\n' >> "${OUT_DIR}/logs/${benchmark_id}.command.txt"

  timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" "${command[@]}" \
    >"${log_path}" 2>&1
  local returncode=$?
  PYTHONPATH=src python3 -m minitrainbench.evidence memory-record \
    --benchmark-id "${benchmark_id}" \
    --config-json "$(printf '{\"tier\":\"%s\",\"strategy\":\"%s\",\"world_size\":%s,\"precision\":\"bf16\",\"vocab_size\":%s,\"d_model\":%s,\"n_heads\":%s,\"n_layers\":%s,\"seq_length\":%s,\"batch_size\":%s,\"activation_checkpointing\":%s}' "${tier}" "${strategy}" "${WORLD_SIZE}" "${vocab_size}" "${d_model}" "${n_heads}" "${n_layers}" "${seq_length}" "${batch_size}" "${ACTIVATION_CHECKPOINTING}")" \
    --command "$(<"${OUT_DIR}/logs/${benchmark_id}.command.txt")" \
    --log "${log_path}" --returncode "${returncode}" --result "${result_path}" \
    --output "${record_path}"

  if [[ ! -f "${record_path}" ]]; then
    echo "未生成结构化记录: ${record_path}" >&2
    return 1
  fi
}

for tier in ${TIERS}; do
  for strategy in ${STRATEGIES}; do
    record="${OUT_DIR}/records/${tier}_${strategy}_${WORLD_SIZE}gpu.json"
    run_case "${tier}" "${strategy}" || true
  done
done

mapfile -t records < <(
  find "${OUT_DIR}/records" -maxdepth 1 -type f \
    -name "*_${WORLD_SIZE}gpu.json" -print | sort
)
if (( ${#records[@]} == 0 )); then
  echo "没有生成显存压力实验记录" >&2
  exit 1
fi

PYTHONPATH=src python3 -m minitrainbench.evidence memory-report \
  --input "${records[@]}" --output "${OUT_DIR}/report.md"

printf '显存压力矩阵结果: %s\n' "${OUT_DIR}/report.md"
