#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-results/evidence_manifest.json}"
mapfile -t inputs < <(
  find results -type f -name '*.json' \
    ! -path '*/rank_*.trace.json' \
    ! -path '*/checkpoints/*' \
    ! -path 'results/evidence_manifest.json' \
    -print | sort
)
if (( ${#inputs[@]} == 0 )); then
  echo "没有可索引的结果 JSON" >&2
  exit 1
fi
PYTHONPATH=src python3 -m minitrainbench.evidence manifest \
  --input "${inputs[@]}" --output "${OUT}"
printf 'Evidence manifest: %s\n' "${OUT}"
