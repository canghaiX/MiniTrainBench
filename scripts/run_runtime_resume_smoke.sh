#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
STRATEGY="${STRATEGY:-fsdp}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
OUT_DIR="${OUT_DIR:-results/runtime_resume}"
KEEP_LAST="${KEEP_LAST:-1}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-128}"
D_MODEL="${D_MODEL:-32}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"

mkdir -p "${OUT_DIR}"

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

checkpoint_dir="${OUT_DIR}/${STRATEGY}_${NPROC}proc_ckpt"
save_output="${OUT_DIR}/${STRATEGY}_${NPROC}proc_save.json"
resume_output="${OUT_DIR}/${STRATEGY}_${NPROC}proc_resume.json"

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
  --warmup-steps 0
  --checkpoint-dir "/workspace/${checkpoint_dir}"
  --save-every 1
  --keep-last "${KEEP_LAST}"
)

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --steps 2 \
  --output "/workspace/${save_output}"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" \
  --resume latest \
  --steps 1 \
  --output "/workspace/${resume_output}"

printf '保存阶段结果: %s\n' "${save_output}"
printf '恢复阶段结果: %s\n' "${resume_output}"
