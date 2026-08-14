#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-results/evidence_manifest.json}"
inputs=()
while IFS= read -r path; do
  if git check-ignore --quiet --no-index "${path}"; then
    continue
  fi
  inputs+=("${path}")
done < <(
  find results -type f -name '*.json' \
    ! -name 'metadata.json' \
    ! -name 'rank_*_summary.json' \
    ! -name 'manifest.json' \
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
