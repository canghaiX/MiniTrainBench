#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
STRATEGY="${STRATEGY:-fsdp}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
OUT_DIR="${OUT_DIR:-results/runtime_determinism}"
KEEP_LAST="${KEEP_LAST:-0}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-128}"
D_MODEL="${D_MODEL:-32}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"

mkdir -p "${OUT_DIR}"

docker_run() {
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" "$@"
}

continuous_dir="${OUT_DIR}/continuous_${STRATEGY}_${NPROC}proc"
interrupted_dir="${OUT_DIR}/interrupted_${STRATEGY}_${NPROC}proc"

common_args=(
  --device "${DEVICE}"
  --strategy "${STRATEGY}"
  --precision "${PRECISION}"
  --batch-size "${BATCH_SIZE}"
  --seq-length "${SEQ_LENGTH}"
  --vocab-size "${VOCAB_SIZE}"
  --d-model "${D_MODEL}"
  --n-heads "${N_HEADS}"
  --n-layers "${N_LAYERS}"
  --dropout "${DROPOUT}"
  --warmup-steps 0
  --repeat 1
  --keep-last "${KEEP_LAST}"
)

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 3 \
  --checkpoint-dir "/workspace/${continuous_dir}" \
  --save-every 3 \
  --output "/workspace/${OUT_DIR}/continuous.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 2 \
  --checkpoint-dir "/workspace/${interrupted_dir}" \
  --save-every 2 \
  --output "/workspace/${OUT_DIR}/interrupted_save.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 1 \
  --checkpoint-dir "/workspace/${interrupted_dir}" \
  --resume latest \
  --save-every 1 \
  --output "/workspace/${OUT_DIR}/interrupted_resume.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench checkpoint verify \
  --device "${DEVICE}" \
  --left "/workspace/${continuous_dir}/step_00000003" \
  --right "/workspace/${interrupted_dir}/step_00000003" \
  --output "/workspace/${OUT_DIR}/verification.json"

printf '精确恢复校验结果: %s\n' "${OUT_DIR}/verification.json"
