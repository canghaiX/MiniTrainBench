#!/usr/bin/env bash

minitrainbench_memory_sample_once() {
  local phase="$1"
  local output="$2"
  local timestamp
  timestamp="$(date +%s.%N)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    while IFS= read -r row; do
      printf '%s,%s,%s\n' "${phase}" "${timestamp}" "${row}" >>"${output}"
    done
}

minitrainbench_memory_sampler_start() {
  local output="$1"
  local pid_name="$2"
  local interval="${3:-0.2}"
  local -n pid_ref="${pid_name}"
  : >"${output}"
  minitrainbench_memory_sample_once baseline "${output}"
  (
    while true; do
      minitrainbench_memory_sample_once sample "${output}"
      sleep "${interval}"
    done
  ) &
  pid_ref=$!
}

minitrainbench_memory_sampler_stop() {
  local pid="$1"
  local output="$2"
  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" 2>/dev/null || true
  minitrainbench_memory_sample_once sample "${output}"
}
