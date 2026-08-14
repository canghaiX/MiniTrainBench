#!/usr/bin/env bash
set -uo pipefail

: "${MEGATRON_DIR:?请设置 MEGATRON_DIR=/path/to/Megatron-LM}"
OUT_DIR="${OUT_DIR:-results/megatron_smoke}"
MATRIX="${MATRIX:-1:1 2:1 4:1 2:2 1:4}"
REPEAT="${REPEAT:-3}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.18.2}"
MEGATRON_IMAGE="${MEGATRON_IMAGE:-minitrainbench:megatron}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
EVIDENCE_MODE="${EVIDENCE_MODE:-formal}"
mkdir -p "${OUT_DIR}/records"

read -r -a matrix_items <<<"${MATRIX}"
aggregates=()
for item in "${matrix_items[@]}"; do
  tp="${item%%:*}"
  pp="${item##*:}"
  name="tp${tp}_pp${pp}"
  rm -f "${OUT_DIR}/records/${name}_trial"*.json "${OUT_DIR}/records/${name}.json"
  trials=()
  for ((trial = 1; trial <= REPEAT; trial++)); do
    MEGATRON_DIR="${MEGATRON_DIR}" MEGATRON_REF="${MEGATRON_REF}" \
      MEGATRON_IMAGE="${MEGATRON_IMAGE}" TRANSFORMER_IMPL="${TRANSFORMER_IMPL}" \
      EVIDENCE_MODE="${EVIDENCE_MODE}" \
      OUT_DIR="${OUT_DIR}" TP="${tp}" PP="${pp}" NAME="${name}" \
      TRIAL_INDEX="${trial}" scripts/run_megatron_smoke.sh || true
    trial_record="${OUT_DIR}/records/${name}_trial$(printf '%02d' "${trial}").json"
    if [[ -f "${trial_record}" ]]; then
      trials+=("${trial_record}")
    fi
  done
  if (( ${#trials[@]} > 0 )); then
    aggregate="${OUT_DIR}/records/${name}.json"
    PYTHONPATH=src python3 -m minitrainbench.evidence megatron-aggregate \
      --input "${trials[@]}" --expected-repeats "${REPEAT}" --output "${aggregate}"
    aggregates+=("${aggregate}")
  fi
done

if (( ${#aggregates[@]} == 0 )); then
  echo "没有生成 Megatron 聚合记录" >&2
  exit 1
fi

PYTHONPATH=src python3 -m minitrainbench.evidence megatron-report \
  --input "${aggregates[@]}" --output "${OUT_DIR}/report.md"

python3 - "${OUT_DIR}" "${REPEAT}" "${#matrix_items[@]}" "${aggregates[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_repeat = int(sys.argv[2])
expected_configs = int(sys.argv[3])
record_paths = [Path(value) for value in sys.argv[4:]]
records = [json.loads(path.read_text()) for path in record_paths]
successful = [row for row in records if row.get("status") == "success"]
all_complete = len(records) == expected_configs and len(successful) == expected_configs
performance_valid = all_complete and all(
    row.get("performance_valid", True) for row in successful
)
invalid_reasons = sorted({
    reason
    for row in records
    for reason in row.get("performance_invalid_reasons", [])
})
manifest = {
    "benchmark": "megatron_matrix",
    "status": (
        "success" if performance_valid else
        "compatibility_smoke" if all_complete else
        "partial" if successful else "failed"
    ),
    "execution_complete": all_complete,
    "performance_valid": performance_valid,
    "performance_invalid_reasons": invalid_reasons,
    "config_count": len(records),
    "successful_config_count": len(successful),
    "expected_config_count": expected_configs,
    "expected_repeat_count": expected_repeat,
    "megatron": records[0].get("megatron"),
    "environment": records[0].get("environment"),
    "provenance": records[0].get("provenance"),
    "records": [
        {
            "path": path.relative_to(out_dir).as_posix(),
            "name": row.get("name"),
            "status": row.get("status"),
            "repeat_count": row.get("repeat_count"),
        }
        for path, row in zip(record_paths, records, strict=True)
    ],
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
sys.exit(0 if all_complete else 1)
PY
status=$?
printf 'Megatron 矩阵结果: %s\n' "${OUT_DIR}/report.md"
exit "${status}"
