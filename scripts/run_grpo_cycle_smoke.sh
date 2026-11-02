#!/usr/bin/env bash
set -euo pipefail

MINI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_ROOT="${INFERKERNELLAB_DIR:-$(cd "${MINI_ROOT}/../InferKernelLab" && pwd)}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
WORLD_SIZE="${WORLD_SIZE:-2}"
KEEP_WORK_DIR="${KEEP_WORK_DIR:-0}"
if [[ -z "${WORK_DIR:-}" ]]; then
  WORK_DIR="$(mktemp -d)"
  if [[ "${KEEP_WORK_DIR}" != "1" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
  fi
else
  KEEP_WORK_DIR=1
fi
mkdir -p "${WORK_DIR}"
if [[ "${DEVICE}" == "cuda" ]]; then
  INFER_DTYPE="${INFER_DTYPE:-bfloat16}"
  TRAIN_PRECISION="${TRAIN_PRECISION:-bf16}"
  DIST_BACKEND="nccl"
else
  INFER_DTYPE="${INFER_DTYPE:-float32}"
  TRAIN_PRECISION="${TRAIN_PRECISION:-fp32}"
  DIST_BACKEND="gloo"
fi
export PYTHONPATH="${MINI_ROOT}/src:${INFER_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"${PYTHON}" -m inferkernellab.rollout \
  --model-source random --export-model-path "${WORK_DIR}/initial-policy" \
  --prompts "${MINI_ROOT}/examples/grpo_prompts.jsonl" \
  --output "${WORK_DIR}/rollouts.jsonl" --device "${DEVICE}" --dtype "${INFER_DTYPE}" \
  --vocab-size 8 --hidden-size 16 --intermediate-size 32 --num-layers 1 \
  --num-attention-heads 4 --num-kv-heads 2 --max-position-embeddings 128 \
  --num-generations 16 --max-new-tokens 8 --temperature 0.8 --top-p 1.0 --seed 2026

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${WORLD_SIZE}" -m minitrainbench rl grpo-train \
  --model-path "${WORK_DIR}/initial-policy" --rollouts "${WORK_DIR}/rollouts.jsonl" \
  --output-model "${WORK_DIR}/updated-policy" --output "${WORK_DIR}/training.json" \
  --device "${DEVICE}" --backend "${DIST_BACKEND}" --precision "${TRAIN_PRECISION}" \
  --steps 2 --groups-per-step 1 --learning-rate 0.0001 --kl-coef 0.01 \
  --checkpoint-dir "${WORK_DIR}/checkpoints" --save-every 1

"${PYTHON}" -m inferkernellab.rollout \
  --model-source path --model-path "${WORK_DIR}/updated-policy" \
  --prompts "${MINI_ROOT}/examples/grpo_prompts.jsonl" \
  --output "${WORK_DIR}/updated-rollouts.jsonl" --device "${DEVICE}" --dtype "${INFER_DTYPE}" \
  --num-generations 16 --max-new-tokens 8 --temperature 0.8 --top-p 1.0 --seed 2027

"${PYTHON}" - "${WORK_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "rollouts.jsonl").read_text().splitlines()[0])
training = json.loads((root / "training.json").read_text())
after = json.loads((root / "updated-rollouts.jsonl").read_text().splitlines()[0])
assert training["status"] == "ok"
assert training["optimizer_updated"]
assert training["dp_parameters_consistent"]
assert before["cache_released"] and after["cache_released"]
assert training["updated_policy_fingerprint"] == after["model_fingerprint"]
assert before["model_fingerprint"] != after["model_fingerprint"]
summary = {
    "schema_version": 1,
    "measurement_scope": "functional smoke without warmup; metrics are not a performance claim",
    "initial_rollout_model_fingerprint": before["model_fingerprint"],
    "updated_rollout_model_fingerprint": after["model_fingerprint"],
    "initial_rollout": before,
    "updated_rollout_manifest": after,
    "training": training,
    "updated_rollout": {
        key: after[key]
        for key in (
            "prompt_group_count", "response_tokens", "decode_tokens_per_sec",
            "prefill_latency_ms", "decode_p50_ms", "decode_p95_ms", "cache_released",
        )
    },
}
destination = Path(os.environ.get("SUMMARY_PATH", root / "summary.json"))
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"summary": str(destination), "optimizer_updated": True}, indent=2))
PY

if [[ "${KEEP_WORK_DIR}" == "1" ]]; then
  echo "Artifacts: ${WORK_DIR}"
fi
