#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source scripts/lib/docker_provenance.sh

: "${MEGATRON_DIR:?请设置 MEGATRON_DIR=/path/to/Megatron-LM}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.18.2}"
MEGATRON_IMAGE="${MEGATRON_IMAGE:-minitrainbench:megatron}"
OUT_DIR="${OUT_DIR:-results/megatron_smoke}"
STAGING_DIR="${STAGING_DIR:-results/.megatron_formal_staging}"
PUBLISH_DIR="${PUBLISH_DIR:-results/.megatron_publish}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
EXPECTED_BASE="nvcr.io/nvidia/pytorch:26.01-py3@sha256:a411b86de9ac003ce5db43894ea7920718512bc02c51a521157c0899aac75631"
EXPECTED_MEGATRON_COMMIT="571370c829ca768fe37244f4e2e7f28d8accc4ab"

gpu_compute_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 !~ /^No/ {seen[$1]=1} END {for (pid in seen) count++; print count + 0}'
}

wait_for_idle_gpus() {
  local started now count
  started="$(date +%s)"
  while true; do
    count="$(gpu_compute_process_count)"
    if (( count == 0 )); then
      printf '8 卡 GPU 已空闲，开始正式 Megatron preflight。\n'
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= WAIT_TIMEOUT_SECONDS )); then
      echo "等待 GPU 空闲超时；未终止任何已有任务。" >&2
      return 1
    fi
    printf '检测到 %s 个已有 GPU 计算进程，%s 秒后重试。\n' \
      "${count}" "${POLL_SECONDS}"
    sleep "${POLL_SECONDS}"
  done
}

if [[ "${OUT_DIR}" == /* || "${STAGING_DIR}" == /* || "${PUBLISH_DIR}" == /* ]]; then
  echo "结果与 staging 目录必须位于当前仓库中" >&2
  exit 2
fi
minitrainbench_assert_source_clean
revision="$(minitrainbench_repo_revision)"
base_image="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}')"
build_revision="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
core_version="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "io.minitrainbench.megatron-core.version"}}')"
nvrx_version="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "io.minitrainbench.nvidia-resiliency-ext.version"}}')"
megatron_commit="$(git -C "${MEGATRON_DIR}" rev-parse HEAD)"
megatron_ref_commit="$(git -C "${MEGATRON_DIR}" rev-parse "${MEGATRON_REF}^{commit}")"

if [[ "${base_image}" != "${EXPECTED_BASE}" ]]; then
  echo "Megatron 镜像没有使用锁定的 NGC 26.01 digest" >&2
  exit 2
fi
if [[ "${build_revision}" != "${revision}" ]]; then
  echo "Megatron 镜像 revision 与当前源码 HEAD 不一致" >&2
  exit 2
fi
if [[ "${core_version}" != "0.18.2" ]]; then
  echo "Megatron Core 版本必须为 0.18.2" >&2
  exit 2
fi
if [[ "${nvrx_version}" != "0.6.0" ]]; then
  echo "nvidia-resiliency-ext 版本必须为 0.6.0" >&2
  exit 2
fi
if [[ "${megatron_commit}" != "${EXPECTED_MEGATRON_COMMIT}" || \
      "${megatron_ref_commit}" != "${EXPECTED_MEGATRON_COMMIT}" ]]; then
  echo "外部 Megatron-LM 必须固定到 core_v0.18.2 对应 commit" >&2
  exit 2
fi
if ! git -C "${MEGATRON_DIR}" diff --quiet || \
   ! git -C "${MEGATRON_DIR}" diff --cached --quiet; then
  echo "外部 Megatron-LM 工作区存在修改，拒绝生成正式结果" >&2
  exit 2
fi

wait_for_idle_gpus

docker run --rm -i --gpus all --ipc=host --network=host \
  -v "${MEGATRON_DIR}:/megatron:ro" -e PYTHONPATH=/megatron \
  -w /megatron "${MEGATRON_IMAGE}" python3 - <<'PY'
import importlib.metadata

import apex
import torch
import transformer_engine
import megatron.core

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert importlib.metadata.version("megatron-core") == "0.18.2"
assert importlib.metadata.version("nvidia-resiliency-ext") == "0.6.0"
print("NGC/TE/Apex/Megatron preflight passed")
PY
wait_for_idle_gpus

rm -rf "${STAGING_DIR}" "${PUBLISH_DIR}"
mkdir -p "${STAGING_DIR}/preflight"
for topology in "1:1" "2:2"; do
  tp="${topology%%:*}"
  pp="${topology##*:}"
  MEGATRON_DIR="${MEGATRON_DIR}" MEGATRON_REF="${MEGATRON_REF}" \
    MEGATRON_IMAGE="${MEGATRON_IMAGE}" OUT_DIR="${STAGING_DIR}/preflight" \
    TP="${tp}" PP="${pp}" NAME="preflight_tp${tp}_pp${pp}" TRIAL_INDEX=1 \
    WARMUP_ITERS=1 MEASURED_ITERS=2 TRANSFORMER_IMPL=auto \
    SEQUENCE_PARALLEL=auto EVIDENCE_MODE=formal TIMEOUT_SECONDS=600 \
    scripts/run_megatron_smoke.sh
done

rm -rf "${STAGING_DIR}"
MEGATRON_DIR="${MEGATRON_DIR}" MEGATRON_REF="${MEGATRON_REF}" \
  MEGATRON_IMAGE="${MEGATRON_IMAGE}" OUT_DIR="${STAGING_DIR}" \
  MATRIX="1:1 2:1 4:1 2:2 1:4" REPEAT=3 WARMUP_ITERS=5 \
  MEASURED_ITERS=20 TRANSFORMER_IMPL=auto SEQUENCE_PARALLEL=auto \
  EVIDENCE_MODE=formal TIMEOUT_SECONDS=1200 \
  scripts/run_megatron_tp_pp_matrix.sh

python3 - "${STAGING_DIR}" "${EXPECTED_BASE}" "${revision}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_base, revision = sys.argv[2:]
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "success", manifest["status"]
assert manifest["execution_complete"] is True
assert manifest["performance_valid"] is True
assert manifest["expected_repeat_count"] == 3
assert manifest["config_count"] == 5

all_records = sorted(root.glob("records/tp*_pp*.json"))
trials = [path for path in all_records if "_trial" in path.stem]
aggregates = [path for path in all_records if "_trial" not in path.stem]
assert len(aggregates) == 5, len(aggregates)
assert len(trials) == 15, len(trials)
for path in trials:
    row = json.loads(path.read_text())
    assert row["status"] == "success", path
    assert row["performance_valid"] is True, path
    assert row["metrics"]["step_sample_count"] == 20, path
    assert row["provenance"]["complete"] is True, path
    assert row["provenance"]["git_revision"] == revision, path
    assert row["provenance"]["base_image"] == expected_base, path
    assert row["megatron"]["resolved_transformer_impl"] == "transformer_engine", path
    assert row["megatron"]["kernel_profile"] == "transformer_engine_fused", path
    assert row["megatron"]["sequence_parallel"] is (row["tp"] > 1), path
    assert row["environment"]["transformer_engine"]["available"] is True, path
    assert row["environment"]["apex"]["available"] is True, path
PY

mkdir -p "${PUBLISH_DIR}"
cp -a "${STAGING_DIR}/records" "${STAGING_DIR}/manifest.json" \
  "${STAGING_DIR}/report.md" "${PUBLISH_DIR}/"
rm -rf "${OUT_DIR}"
mv "${PUBLISH_DIR}" "${OUT_DIR}"
printf '正式 Megatron 8 卡矩阵已发布到 %s\n' "${OUT_DIR}"
