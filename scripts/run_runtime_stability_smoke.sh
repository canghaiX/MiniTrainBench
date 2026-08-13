#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
STRATEGY="${STRATEGY:-fsdp}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
OUT_DIR="${OUT_DIR:-results/runtime_stability}"
KEEP_LAST="${KEEP_LAST:-0}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-128}"
D_MODEL="${D_MODEL:-32}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-6}"
MIN_LR="${MIN_LR:-0.00003}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
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
  --lr-scheduler cosine
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-decay-steps "${LR_DECAY_STEPS}"
  --min-learning-rate "${MIN_LR}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --repeat 1
  --keep-last "${KEEP_LAST}"
)

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 6 \
  --checkpoint-dir "/workspace/${continuous_dir}" \
  --save-every 6 \
  --output "/workspace/${OUT_DIR}/continuous.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 3 \
  --checkpoint-dir "/workspace/${interrupted_dir}" \
  --save-every 3 \
  --output "/workspace/${OUT_DIR}/interrupted_save.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 3 \
  --checkpoint-dir "/workspace/${interrupted_dir}" \
  --resume latest \
  --save-every 3 \
  --output "/workspace/${OUT_DIR}/interrupted_resume.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench checkpoint verify \
  --device "${DEVICE}" \
  --left "/workspace/${continuous_dir}/step_00000006" \
  --right "/workspace/${interrupted_dir}/step_00000006" \
  --output "/workspace/${OUT_DIR}/verification.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench fault smoke \
  --device "${DEVICE}" \
  --backend nccl \
  --strategy "${STRATEGY}" \
  --precision "${PRECISION}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --vocab-size "${VOCAB_SIZE}" \
  --d-model "${D_MODEL}" \
  --n-heads "${N_HEADS}" \
  --n-layers "${N_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr-scheduler cosine \
  --lr-warmup-steps 1 \
  --lr-decay-steps 3 \
  --min-learning-rate "${MIN_LR}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --continuous-steps 2 \
  --interrupted-steps 1 \
  --resume-steps 1 \
  --keep-last "${KEEP_LAST}" \
  --output "/workspace/${OUT_DIR}/fault/fault_tolerance.json"

docker_run python3 -m minitrainbench report \
  --input "/workspace/${OUT_DIR}/continuous.json" \
  "/workspace/${OUT_DIR}/interrupted_resume.json" \
  "/workspace/${OUT_DIR}/fault/fault_tolerance.json" \
  "/workspace/${OUT_DIR}/verification.json" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'Runtime stability 结果: %s/report.md\n' "${OUT_DIR}"
