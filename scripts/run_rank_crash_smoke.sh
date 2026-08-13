#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/docker_provenance.sh"

IMAGE="${IMAGE:-minitrainbench:gpu}"
NPROC="${NPROC:-2}"
STRATEGY="${STRATEGY:-fsdp}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
OUT_DIR="${OUT_DIR:-results/rank_crash}"
CRASH_RANK="${CRASH_RANK:-1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-2}"
FINAL_STEP="${FINAL_STEP:-4}"

mkdir -p "${OUT_DIR}"
reference_dir="${OUT_DIR}/reference"
interrupted_dir="${OUT_DIR}/interrupted"
rm -rf "${reference_dir}" "${interrupted_dir}"

common_args=(
  --device "${DEVICE}" --strategy "${STRATEGY}" --precision "${PRECISION}"
  --batch-size 1 --seq-length 16 --vocab-size 128
  --d-model 32 --n-heads 4 --n-layers 1 --dropout 0.1
  --warmup-steps 0 --repeat 1 --keep-last 0
  --lr-scheduler cosine --lr-warmup-steps 1
  --lr-decay-steps "${FINAL_STEP}" --min-learning-rate 0.00003
  --max-grad-norm 1.0
)

docker_run() {
  minitrainbench_docker_run "${IMAGE}" "$@"
}

checkpoint_digest() {
  local root="$1"
  (
    cd "${root}"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
  )
}

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" --steps "${FINAL_STEP}" \
  --checkpoint-dir "/workspace/${reference_dir}" --save-every "${FINAL_STEP}" \
  --output "/workspace/${OUT_DIR}/reference.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" --steps "${CHECKPOINT_STEP}" \
  --checkpoint-dir "/workspace/${interrupted_dir}" --save-every "${CHECKPOINT_STEP}" \
  --output "/workspace/${OUT_DIR}/interrupted_seed.json"

before_digest="$(checkpoint_digest "${interrupted_dir}")"
set +e
docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench \
  fault crash-worker "${common_args[@]}" --steps "$((FINAL_STEP - CHECKPOINT_STEP))" \
  --checkpoint-dir "/workspace/${interrupted_dir}" --resume latest \
  --crash-rank "${CRASH_RANK}" --crash-at-step "${CHECKPOINT_STEP}" \
  --output "/workspace/${OUT_DIR}/unexpected_crash_output.json" \
  >"${OUT_DIR}/crash.log" 2>&1
crash_returncode=$?
set -e
after_digest="$(checkpoint_digest "${interrupted_dir}")"

if [[ "${crash_returncode}" -eq 0 ]]; then
  echo "rank crash 命令意外成功" >&2
  exit 1
fi
if [[ "${before_digest}" != "${after_digest}" ]]; then
  echo "rank crash 后 READY checkpoint 发生变化" >&2
  exit 1
fi

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench train \
  "${common_args[@]}" --steps "$((FINAL_STEP - CHECKPOINT_STEP))" \
  --checkpoint-dir "/workspace/${interrupted_dir}" --resume latest \
  --save-every "${FINAL_STEP}" --output "/workspace/${OUT_DIR}/resumed.json"

docker_run torchrun --standalone --nproc_per_node="${NPROC}" -m minitrainbench \
  checkpoint verify --device "${DEVICE}" \
  --left "/workspace/${reference_dir}/step_$(printf '%08d' "${FINAL_STEP}")" \
  --right "/workspace/${interrupted_dir}/step_$(printf '%08d' "${FINAL_STEP}")" \
  --output "/workspace/${OUT_DIR}/verification.json"

python3 - "${OUT_DIR}" "${crash_returncode}" "${CRASH_RANK}" \
  "${CHECKPOINT_STEP}" "${before_digest}" "${after_digest}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
returncode = int(sys.argv[2])
crash_rank = int(sys.argv[3])
checkpoint_step = int(sys.argv[4])
before_digest, after_digest = sys.argv[5:7]
resumed = json.loads((root / "resumed.json").read_text())
verification = json.loads((root / "verification.json").read_text())
payload = {
    "benchmark": "fault_tolerance",
    "strategy": resumed["strategy"],
    "world_size": resumed["world_size"],
    "precision": resumed["precision"],
    "environment": resumed["environment"],
    "provenance": resumed["provenance"],
    "recovery_mode": "manual_restart",
    "crash_log_sha256": hashlib.sha256((root / "crash.log").read_bytes()).hexdigest(),
    "verification": verification,
    "failure_handling": [{
        "failure_type": "rank_crash",
        "detection": "torchrun worker group non-zero exit",
        "auto_recovered": False,
        "recovery_mode": "manual_restart",
        "crash_rank": crash_rank,
        "crash_at_step": checkpoint_step,
        "launcher_returncode": returncode,
        "checkpoint_digest_before": before_digest,
        "checkpoint_digest_after": after_digest,
        "checkpoint_unchanged": before_digest == after_digest,
        "recovered_checkpoint": resumed["runtime"]["resume_path"],
        "global_step": resumed["global_step"],
        "tokens_seen": resumed["tokens_seen"],
        "status": "recovered_exact" if verification["exact_match"] else "failed",
    }],
}
(root / "rank_crash.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY

docker_run python3 -m minitrainbench report \
  --input "/workspace/${OUT_DIR}/rank_crash.json" \
  --output "/workspace/${OUT_DIR}/report.md"

printf 'Rank crash 恢复证据: %s/report.md\n' "${OUT_DIR}"
