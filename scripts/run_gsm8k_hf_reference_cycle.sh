#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/data/models/Qwen/Qwen3-0.6B}"
: "${INFER_ROOT:=../InferKernelLab}"
: "${OUT_ROOT:=results/gsm8k_hf_reference_single_seed}"
: "${DATA_SOURCE_DIR:=results/gsm8k_grpo/data}"
: "${TRAIN_GPUS:=0,1}"
: "${HF_REFERENCE_GPU:=0}"
: "${SERVER_GPU:=7}"
: "${SGLANG_IMAGE:=harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.2.0-cuda-sglang0.5.18-ci}"
: "${PORT:=30000}"
: "${SEEDS:=2026}"
: "${ROLLOUT_MAX_NEW_TOKENS:=256}"
: "${SMOKE_GROUPS:=8}"
: "${SMOKE_MAX_NEW_TOKENS:=256}"

evaluate_only=false
if [[ "${1:-}" == --evaluate-only && $# == 1 ]]; then
  evaluate_only=true
elif (( $# != 0 )); then
  echo "Usage: $0 [--evaluate-only]" >&2
  exit 2
fi

OUT_ROOT="$(realpath -m "$OUT_ROOT")"
INFER_ROOT="$(realpath "$INFER_ROOT")"
DATA_DIR="$OUT_ROOT/data"
SMOKE_DIR="$OUT_ROOT/smoke"
CONTAINER_NAME="minitrainbench-hf-reference-$$"
if [[ "$evaluate_only" == false && -e "$OUT_ROOT" ]]; then
  echo "Use a new OUT_ROOT to preserve existing experiment artifacts: $OUT_ROOT" >&2
  exit 1
fi
if [[ "$evaluate_only" == true && ! -d "$OUT_ROOT" ]]; then
  echo "Evaluation continuation requires an existing OUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT" "$SMOKE_DIR"
exec > >(tee -a "$OUT_ROOT/cycle.log") 2>&1
stage="setup"
trap 'printf "stage=%s line=%s exit=%s\n" "$stage" "$LINENO" "$?" > "$OUT_ROOT/failure.txt"' ERR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cleanup() {
  docker logs --tail 2000 "$CONTAINER_NAME" > "$OUT_ROOT/server_last.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$evaluate_only" == false ]]; then
PYTHONPATH=src python3 scripts/prepare_gsm8k.py \
  --output-dir "$DATA_DIR" --source-dir "$DATA_SOURCE_DIR" --offline \
  --train-samples 256 --seed 2026
head -n "$SMOKE_GROUPS" "$DATA_DIR/train.jsonl" > "$SMOKE_DIR/prompts.jsonl"

run_rollout() {
  local prompts="$1" output="$2" seed="$3" max_tokens="$4"
  CUDA_VISIBLE_DEVICES="$HF_REFERENCE_GPU" PYTHONPATH="$INFER_ROOT/src" \
    python3 -m inferkernellab.rollout \
      --engine hf_reference --model-source path --model-path "$MODEL_PATH" \
      --tokenizer-path "$MODEL_PATH" --data-manifest "$DATA_DIR/manifest.json" \
      --prompts "$prompts" --output "$output" --device cuda --dtype bfloat16 \
      --attention-implementation sdpa --num-generations 8 \
      --max-new-tokens "$max_tokens" --temperature 0.8 --top-p 1.0 --seed "$seed"
}

run_train() {
  local rollouts="$1" output_model="$2" report="$3" steps="$4" checkpoint="$5"
  shift 5
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=src \
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
      -m minitrainbench rl grpo-train \
      --model-path "$MODEL_PATH" --rollouts "$rollouts" \
      --output-model "$output_model" --output "$report" --device cuda \
      --backend nccl --precision bf16 --attention-implementation sdpa \
      --steps "$steps" --learning-rate 1e-6 --clip-epsilon 0.2 --kl-coef 0.01 \
      --checkpoint-dir "$checkpoint" --save-every 1 --keep-last 2 "$@"
}

stage="smoke"
echo "$(date -Is) stage=$stage"
run_rollout "$SMOKE_DIR/prompts.jsonl" "$SMOKE_DIR/rollout.jsonl" \
  2026 "$SMOKE_MAX_NEW_TOKENS"
python3 - "$SMOKE_DIR/rollout.jsonl" <<'PY'
import json
import sys

groups = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if json.loads(line).get("type") == "prompt_group"
]
varying = [
    group["group_id"]
    for group in groups
    if len({completion["reward"] for completion in group["completions"]}) > 1
]
if not varying:
    raise SystemExit("HF update smoke has no within-group reward variance")
print(f"reward-bearing smoke groups: {varying}")
PY
run_train "$SMOKE_DIR/rollout.jsonl" "$SMOKE_DIR/updated-step1" \
  "$SMOKE_DIR/preflight.json" 1 "$SMOKE_DIR/checkpoints" \
  --groups-per-step 1 --microbatch-groups 1 --preflight-only
python3 - "$SMOKE_DIR/preflight.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
probe = report.get("logprob_probe", {})
if report.get("status") != "ok" or not probe.get("passed"):
    raise SystemExit("A100 HF reference smoke preflight failed")
if probe.get("max_abs_error", float("inf")) > 0.5:
    raise SystemExit("A100 HF reference smoke exceeded the 0.5 logprob gate")
PY
run_train "$SMOKE_DIR/rollout.jsonl" "$SMOKE_DIR/updated-step1" \
  "$SMOKE_DIR/train-step1.json" 1 "$SMOKE_DIR/checkpoints" \
  --groups-per-step 1 --microbatch-groups 1
python3 - "$SMOKE_DIR/train-step1.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("optimizer_updated"):
    raise SystemExit("A100 HF reference smoke did not update parameters")
if report.get("initial_policy_fingerprint") == report.get("updated_policy_fingerprint"):
    raise SystemExit("A100 HF reference smoke policy fingerprint did not change")
if not report.get("dp_parameters_consistent"):
    raise SystemExit("A100 HF reference smoke has inconsistent DDP parameters")
PY
run_train "$SMOKE_DIR/rollout.jsonl" "$SMOKE_DIR/updated-resumed" \
  "$SMOKE_DIR/train-resumed.json" 2 "$SMOKE_DIR/checkpoints" \
  --groups-per-step 1 --microbatch-groups 1 --resume latest
python3 - "$SMOKE_DIR/train-resumed.json" <<'PY'
import json
import sys

resumed = json.load(open(sys.argv[1], encoding="utf-8"))
if not resumed.get("resumed") or resumed.get("global_step") != 2:
    raise SystemExit("A100 HF reference checkpoint did not resume to step 2")
if resumed.get("updated_policy_fingerprint") == resumed.get("initial_policy_fingerprint"):
    raise SystemExit("A100 HF reference resumed policy does not contain the saved update")
if not resumed.get("dp_parameters_consistent"):
    raise SystemExit("A100 HF reference resumed update failed")
print("A100 HF reference DCP resume reached step 2 with consistent updated parameters")
PY
python3 scripts/cleanup_gsm8k_artifacts.py --root "$OUT_ROOT" --phase smoke

for seed in $SEEDS; do
  stage="rollout_seed_$seed"
  echo "$(date -Is) stage=$stage"
  seed_root="$OUT_ROOT/seed_$seed"
  mkdir -p "$seed_root"
  run_rollout "$DATA_DIR/train.jsonl" "$seed_root/rollout.jsonl" \
    "$seed" "$ROLLOUT_MAX_NEW_TOKENS"
done

for seed in $SEEDS; do
  stage="preflight_seed_$seed"
  echo "$(date -Is) stage=$stage"
  seed_root="$OUT_ROOT/seed_$seed"
  run_train "$seed_root/rollout.jsonl" "$seed_root/updated_model" \
    "$seed_root/preflight.json" 16 "$seed_root/checkpoints" \
    --groups-per-step 8 --microbatch-groups 1 --preflight-only --seed "$seed"
  python3 - "$seed_root/preflight.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
probe = report.get("logprob_probe", {})
error = probe.get("max_abs_error")
if report.get("status") != "ok" or not probe.get("passed"):
    raise SystemExit("full HF reference policy preflight failed")
if not isinstance(error, (int, float)) or not math.isfinite(error) or error > 0.5:
    raise SystemExit("full HF reference policy preflight exceeded max error 0.5")
PY
done

for seed in $SEEDS; do
  stage="train_seed_$seed"
  echo "$(date -Is) stage=$stage"
  seed_root="$OUT_ROOT/seed_$seed"
  run_train "$seed_root/rollout.jsonl" "$seed_root/updated_model" \
    "$seed_root/train.json" 16 "$seed_root/checkpoints" \
    --groups-per-step 8 --microbatch-groups 1 --seed "$seed" \
    --preflight-report "$seed_root/preflight.json"
  python3 - "$seed_root/train.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "ok" or report.get("global_step") != 16:
    raise SystemExit("formal HF reference training did not complete all 16 steps")
if not report.get("optimizer_updated") or not report.get("dp_parameters_consistent"):
    raise SystemExit("formal HF reference training did not produce a consistent update")
if not report.get("logprob_probe", {}).get("passed"):
    raise SystemExit("formal HF reference training report has a failed policy probe")
PY
done
fi

# Also validate an interrupted run before continuing with evaluation. This mode
# never regenerates rollout data or starts another optimizer update.
stage="verify_training"
for seed in $SEEDS; do
  python3 - "$OUT_ROOT/seed_$seed/train.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
probe = report.get("logprob_probe", {})
error = probe.get("max_abs_error")
if (
    report.get("status") != "ok"
    or report.get("global_step") != 16
    or report.get("optimizer_updated") is not True
    or report.get("dp_parameters_consistent") is not True
    or report.get("initial_policy_fingerprint") == report.get("updated_policy_fingerprint")
    or probe.get("passed") is not True
    or type(error) not in (int, float)
    or not math.isfinite(error)
    or not 0 <= error <= 0.5
):
    raise SystemExit("evaluation requires verified 16-step training and a passed full gate")
PY
done

start_sglang() {
  local model_path="$1"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/model_info" >/dev/null 2>&1; then
    echo "SGLang port $PORT is already owned by another server" >&2
    return 1
  fi
  docker run -d --name "$CONTAINER_NAME" --gpus "device=$SERVER_GPU" \
    --ipc=host --shm-size=16g --network=host \
    --log-opt max-size=10m --log-opt max-file=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v /data/models:/data/models:ro -v "$OUT_ROOT:/experiment:ro" \
    "$SGLANG_IMAGE" python3 -m sglang.launch_server \
      --model-path "$model_path" --host 0.0.0.0 --port "$PORT" --tp-size 1 \
      --dtype bfloat16 --served-model-name qwen3-0.6b --disable-cuda-graph \
      --rl-on-policy-target fsdp --attention-backend fa3 >/dev/null
  for _ in $(seq 1 120); do
    if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/model_info" >/dev/null 2>&1; then
      return
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != true ]]; then
      docker logs "$CONTAINER_NAME" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "SGLang serving endpoint did not become ready" >&2
  return 1
}

docker image inspect "$SGLANG_IMAGE" --format '{{json .Id}}' \
  > "$OUT_ROOT/sglang_image_id.json"
stage="eval_initial"
echo "$(date -Is) stage=$stage"
start_sglang "$MODEL_PATH"
PYTHONPATH="$INFER_ROOT/src" python3 "$INFER_ROOT/scripts/evaluate_gsm8k.py" \
  --model-path "$MODEL_PATH" --data "$DATA_DIR/test.jsonl" \
  --base-url "http://127.0.0.1:$PORT" --output "$OUT_ROOT/eval_initial.json" \
  --batch-size 128 --max-new-tokens 256

for seed in $SEEDS; do
  stage="eval_seed_$seed"
  echo "$(date -Is) stage=$stage"
  start_sglang "/experiment/seed_$seed/updated_model"
  PYTHONPATH="$INFER_ROOT/src" python3 "$INFER_ROOT/scripts/evaluate_gsm8k.py" \
    --model-path "$OUT_ROOT/seed_$seed/updated_model" \
    --data "$DATA_DIR/test.jsonl" --base-url "http://127.0.0.1:$PORT" \
    --output "$OUT_ROOT/eval_seed_$seed.json" --batch-size 128 --max-new-tokens 256
done

stage="summary"
PYTHONPATH=src python3 scripts/summarize_gsm8k_grpo.py \
  --root "$OUT_ROOT" --output "$OUT_ROOT/summary.json" \
  --expected-seeds "$(wc -w <<< "$SEEDS")"
python3 scripts/cleanup_gsm8k_artifacts.py --root "$OUT_ROOT" --phase complete
echo "$(date -Is) stage=complete"
