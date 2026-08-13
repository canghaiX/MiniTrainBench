#!/usr/bin/env bash

minitrainbench_repo_revision() {
  git rev-parse HEAD
}

minitrainbench_assert_source_clean() {
  local dirty
  dirty="$(
    git status --porcelain --untracked-files=normal -- . \
      ':(exclude)results' ':(exclude)results/**'
  )"
  if [[ -n "${dirty}" && "${ALLOW_UNVERIFIED_PROVENANCE:-0}" != "1" ]]; then
    echo "源码工作区存在未提交修改，拒绝生成正式 benchmark 证据：" >&2
    printf '%s\n' "${dirty}" >&2
    echo "仅调试时可设置 ALLOW_UNVERIFIED_PROVENANCE=1。" >&2
    return 2
  fi
}

minitrainbench_public_image_ref() {
  local image="$1"
  if [[ "${image}" == *.*/* ]] && \
     [[ "${image}" != docker.io/* ]] && \
     [[ "${image}" != pytorch/* ]] && \
     [[ "${image}" != nvcr.io/* ]]; then
    printf 'redacted'
  else
    printf '%s' "${image}"
  fi
}

minitrainbench_docker_command() {
  local image="$1"
  local output_name="$2"
  shift 2
  local -n output="${output_name}"
  local revision image_id build_revision base_image image_ref inner_command
  local -a extra_args=()

  minitrainbench_assert_source_clean
  revision="$(minitrainbench_repo_revision)"
  image_id="$(docker image inspect "${image}" --format '{{.Id}}')"
  build_revision="$(docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  base_image="$(docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}')"
  image_ref="$(minitrainbench_public_image_ref "${image}")"
  printf -v inner_command '%q ' "$@"
  if declare -p MINITRAINBENCH_DOCKER_ARGS >/dev/null 2>&1; then
    extra_args=("${MINITRAINBENCH_DOCKER_ARGS[@]}")
  fi

  if [[ "${build_revision}" != "${revision}" && \
        "${ALLOW_UNVERIFIED_PROVENANCE:-0}" != "1" ]]; then
    echo "镜像 build revision (${build_revision}) 与源码 HEAD (${revision}) 不一致。" >&2
    echo "请先运行 scripts/build_images.sh；仅调试时可设置 ALLOW_UNVERIFIED_PROVENANCE=1。" >&2
    return 2
  fi
  if [[ "${image_ref}" == "redacted" && \
        "${ALLOW_UNVERIFIED_PROVENANCE:-0}" != "1" ]]; then
    echo "私有 registry 镜像不能生成公开 benchmark 证据。" >&2
    return 2
  fi

  output=(
    docker run --rm --gpus all --ipc=host --network=host
    -v "${PWD}:/workspace" -w /workspace
    -e "MINITRAINBENCH_GIT_REVISION=${revision}"
    -e MINITRAINBENCH_GIT_DIRTY=false
    -e "MINITRAINBENCH_IMAGE_REF=${image_ref}"
    -e "MINITRAINBENCH_IMAGE_ID=${image_id}"
    -e "MINITRAINBENCH_BASE_IMAGE=${base_image}"
    -e "MINITRAINBENCH_BUILD_REVISION=${build_revision}"
    -e "MINITRAINBENCH_COMMAND=${inner_command% }"
    "${extra_args[@]}"
    "${image}" "$@"
  )
}

minitrainbench_docker_run() {
  local image="$1"
  shift
  local -a command
  minitrainbench_docker_command "${image}" command "$@"
  "${command[@]}"
}
