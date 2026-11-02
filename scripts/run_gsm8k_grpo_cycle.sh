#!/usr/bin/env bash
set -euo pipefail

# The server owns one GPU; torchrun uses the GPUs listed in TRAIN_GPUS.
: "${SGLANG_IMAGE:=harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.2.0-cuda-sglang0.5.18-ci}"
: "${MODEL_PATH:=/data/models/Qwen/Qwen3-0.6B}"
: "${INFER_ROOT:=../InferKernelLab}"
: "${OUT_ROOT:=results/gsm8k_grpo}"
: "${SERVER_GPU:=7}"
: "${TRAIN_GPUS:=0,1}"
: "${PORT:=30000}"
: "${SEEDS:=2026 2027 2028}"
: "${ROLLOUT_MAX_NEW_TOKENS:=256}"
: "${SGLANG_ATTENTION_BACKEND:=fa3}"

OUT_ROOT="$(realpath -m "$OUT_ROOT")"
DATA_DIR="$OUT_ROOT/data"
mkdir -p "$OUT_ROOT"
PYTHONPATH=src python3 scripts/prepare_gsm8k.py --output-dir "$DATA_DIR" --train-samples 256 --seed 2026

container_name="minitrainbench-sglang-gsm8k-$$"
cleanup() {
  docker logs "$container_name" > "$OUT_ROOT/server_last.log" 2>&1 || true
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
start_server() {
  local model_path="$1"
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  docker run -d --name "$container_name" --gpus "device=$SERVER_GPU" --ipc=host --shm-size=16g --network=host \
    -v /data/models:/data/models:ro -v "$OUT_ROOT:/experiment:ro" "$SGLANG_IMAGE" \
    python3 -m sglang.launch_server --model-path "$model_path" --host 0.0.0.0 --port "$PORT" \
    --tp-size 1 --dtype bfloat16 --served-model-name qwen3-0.6b --disable-cuda-graph --rl-on-policy-target fsdp \
    --attention-backend "$SGLANG_ATTENTION_BACKEND" >/dev/null
  for _ in $(seq 1 120); do
    if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/model_info" >/dev/null 2>&1; then return; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" != true ]]; then
      docker logs "$container_name" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "SGLang server did not become ready" >&2
  exit 1
}
start_server "$MODEL_PATH"
docker image inspect "$SGLANG_IMAGE" --format '{{json .Id}}' > "$OUT_ROOT/server_image_id.json"
docker exec "$container_name" python3 -c \
  'import json, torch, transformers, sglang; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "sglang": sglang.__version__}, indent=2))' \
  > "$OUT_ROOT/server_runtime.json"

for seed in $SEEDS; do
  seed_root="$OUT_ROOT/seed_$seed"
  mkdir -p "$seed_root"
  PYTHONPATH="$INFER_ROOT/src" python3 -m inferkernellab.rollout \
    --engine sglang --sglang-base-url "http://127.0.0.1:$PORT" --sglang-timeout 600 \
    --model-source path --model-path "$MODEL_PATH" --prompts "$DATA_DIR/train.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" --output "$seed_root/rollout.jsonl" \
    --dtype bfloat16 --num-generations 8 --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS" --temperature 0.8 --top-p 1.0 --seed "$seed"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=src python3 -m torch.distributed.run \
    --standalone --nproc_per_node=2 -m minitrainbench rl grpo-train \
    --model-path "$MODEL_PATH" --rollouts "$seed_root/rollout.jsonl" \
    --output-model "$seed_root/updated_model" --output "$seed_root/train.json" \
    --device cuda --backend nccl --precision bf16 --steps 16 --groups-per-step 8 --microbatch-groups 1 \
    --attention-implementation sdpa \
    --learning-rate 1e-6 --clip-epsilon 0.2 --kl-coef 0.01 --seed "$seed" \
    --logprob-tolerance 0.5 \
    --checkpoint-dir "$seed_root/checkpoints" --save-every 8
done

models=("$MODEL_PATH")
seed_count=0
for seed in $SEEDS; do
  models+=("$OUT_ROOT/seed_$seed/updated_model")
  seed_count=$((seed_count + 1))
done
for model in "${models[@]}"; do
  [[ -d "$model" ]] || continue
  if [[ "$model" == "$MODEL_PATH" ]]; then
    label=initial
    server_model="$model"
  else
    label="$(basename "$(dirname "$model")")"
    server_model="/experiment/${model#"$OUT_ROOT/"}"
  fi
  start_server "$server_model"
  host_model="$model"
  PYTHONPATH="$INFER_ROOT/src" python3 "$INFER_ROOT/scripts/evaluate_gsm8k.py" \
    --model-path "$host_model" --data "$DATA_DIR/test.jsonl" \
    --base-url "http://127.0.0.1:$PORT" --output "$OUT_ROOT/eval_${label}.json" \
    --batch-size 128 --max-new-tokens 256
done

PYTHONPATH=src python3 scripts/summarize_gsm8k_grpo.py \
  --root "$OUT_ROOT" --output "$OUT_ROOT/summary.json" --expected-seeds "$seed_count"
