#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
OUT_DIR="${OUT_DIR:-results/profiler}"
PROFILE_WAIT="${PROFILE_WAIT:-1}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
RECORD_SHAPES="${RECORD_SHAPES:-0}"
WITH_STACK="${WITH_STACK:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"

if [[ "${ALLOW_BUSY_GPUS}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy_gpus="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$2 + 0 > 1024 {gsub(/ /, "", $1); print $1}' \
      | paste -sd, -
  )"
  if [[ -n "${busy_gpus}" ]]; then
    echo "GPU ${busy_gpus} 已有超过 1024 MB 显存占用；拒绝污染 Profiler 证据。" >&2
    echo "确认允许并发采样时设置 ALLOW_BUSY_GPUS=1。" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_DIR}"

docker_run() {
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" "$@"
}

run_profile() {
  local strategy="$1"
  local trace_dir="${OUT_DIR}/${strategy}_${NPROC}gpu"
  docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench profile \
    --strategy "${strategy}" \
    --precision bf16 \
    --batch-size "${BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --vocab-size "${VOCAB_SIZE}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --profile-wait "${PROFILE_WAIT}" \
    --profile-warmup "${PROFILE_WARMUP}" \
    --profile-active "${PROFILE_ACTIVE}" \
    --trace-dir "/workspace/${trace_dir}" \
    $(if [[ "${RECORD_SHAPES}" == "1" ]]; then echo --record-shapes; fi) \
    $(if [[ "${WITH_STACK}" == "1" ]]; then echo --with-stack; fi)
}

run_profile ddp
run_profile fsdp

python3 - <<PY
import json
from pathlib import Path

out_dir = Path("${OUT_DIR}")
trace_dirs = [
    out_dir / f"ddp_${NPROC}gpu",
    out_dir / f"fsdp_${NPROC}gpu",
]
parts = ["# MiniTrainBench Profiler 汇总", ""]
payload = {
    "benchmark": "profile_matrix",
    "nproc": int("${NPROC}"),
    "profiles": [],
}
for trace_dir in trace_dirs:
    summary = trace_dir / "profile_summary.md"
    if summary.is_file():
        parts.append(f"## {trace_dir.name}")
        parts.append("")
        parts.append(summary.read_text())
        parts.append("")
    summary_json = trace_dir / "profile_summary.json"
    if summary_json.is_file():
        payload["profiles"].append(json.loads(summary_json.read_text()))
summary_md = out_dir / "profile_summary.md"
summary_json = out_dir / "profile_summary.json"
report = out_dir / "report.md"
summary_text = "\n".join(parts).rstrip() + "\n"
summary_md.write_text(summary_text)
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
report.write_text(summary_text)
print(summary_md)
PY

printf 'Profiler 汇总结果: %s\n' "${OUT_DIR}/profile_summary.md"
