#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source scripts/lib/docker_provenance.sh

GPU_IMAGE="${GPU_IMAGE:-minitrainbench:gpu}"
DEEPSPEED_IMAGE="${DEEPSPEED_IMAGE:-minitrainbench:deepspeed}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b}"
SOURCE_URL="${SOURCE_URL:-https://github.com/canghaiX/MiniTrainBench}"
BUILD_DEEPSPEED="${BUILD_DEEPSPEED:-1}"

minitrainbench_assert_source_clean
revision="$(minitrainbench_repo_revision)"
build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
common_args=(
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "VCS_REF=${revision}"
  --build-arg "BUILD_DATE=${build_date}"
  --build-arg "SOURCE_URL=${SOURCE_URL}"
)
secret_args=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  secret_args+=(--secret id=https_proxy,env=HTTPS_PROXY)
fi

docker build --target gpu "${common_args[@]}" -t "${GPU_IMAGE}" .
if [[ "${BUILD_DEEPSPEED}" == "1" ]]; then
  docker build --target gpu-deepspeed "${common_args[@]}" \
    "${secret_args[@]}" -t "${DEEPSPEED_IMAGE}" .
fi

printf 'GPU image: %s (%s)\n' "${GPU_IMAGE}" "$(docker image inspect "${GPU_IMAGE}" --format '{{.Id}}')"
if [[ "${BUILD_DEEPSPEED}" == "1" ]]; then
  printf 'DeepSpeed image: %s (%s)\n' "${DEEPSPEED_IMAGE}" "$(docker image inspect "${DEEPSPEED_IMAGE}" --format '{{.Id}}')"
fi
