#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source scripts/lib/docker_provenance.sh

IMAGE="${IMAGE:-minitrainbench:megatron}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3@sha256:a411b86de9ac003ce5db43894ea7920718512bc02c51a521157c0899aac75631}"
SOURCE_URL="${SOURCE_URL:-https://github.com/canghaiX/MiniTrainBench}"
MEGATRON_CORE_VERSION="${MEGATRON_CORE_VERSION:-0.18.2}"

minitrainbench_assert_source_clean
revision="$(minitrainbench_repo_revision)"
build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
secret_args=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  secret_args+=(--secret id=https_proxy,env=HTTPS_PROXY)
fi

docker build -f Dockerfile.megatron \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VCS_REF=${revision}" \
  --build-arg "BUILD_DATE=${build_date}" \
  --build-arg "SOURCE_URL=${SOURCE_URL}" \
  --build-arg "MEGATRON_CORE_VERSION=${MEGATRON_CORE_VERSION}" \
  "${secret_args[@]}" -t "${IMAGE}" .

printf 'Megatron image: %s (%s)\n' \
  "${IMAGE}" "$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
