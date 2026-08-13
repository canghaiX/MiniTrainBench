#!/usr/bin/env bash
set -uo pipefail

: "${MEGATRON_DIR:?请设置 MEGATRON_DIR=/path/to/Megatron-LM}"
OUT_DIR="${OUT_DIR:-results/megatron_smoke}"
MATRIX="${MATRIX:-1:1 2:1 4:1 2:2 1:4}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.18.2}"
MEGATRON_IMAGE="${MEGATRON_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
mkdir -p "${OUT_DIR}/records"

records=()
for item in ${MATRIX}; do
  tp="${item%%:*}"
  pp="${item##*:}"
  name="tp${tp}_pp${pp}"
  MEGATRON_DIR="${MEGATRON_DIR}" MEGATRON_REF="${MEGATRON_REF}" \
    MEGATRON_IMAGE="${MEGATRON_IMAGE}" TRANSFORMER_IMPL="${TRANSFORMER_IMPL}" \
    OUT_DIR="${OUT_DIR}" TP="${tp}" PP="${pp}" \
    NAME="${name}" scripts/run_megatron_smoke.sh || true
  record="${OUT_DIR}/records/${name}.json"
  if [[ -f "${record}" ]]; then
    records+=("${record}")
  fi
done

if (( ${#records[@]} == 0 )); then
  echo "没有生成 Megatron 记录" >&2
  exit 1
fi

PYTHONPATH=src python3 -m minitrainbench.evidence megatron-report \
  --input "${records[@]}" --output "${OUT_DIR}/report.md"

python3 - "${OUT_DIR}" "${MEGATRON_DIR}" "${MEGATRON_REF}" "${MEGATRON_IMAGE}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
megatron_dir = Path(sys.argv[2])
megatron_ref = sys.argv[3]
megatron_image = sys.argv[4]
records = [json.loads(path.read_text()) for path in sorted((out_dir / "records").glob("*.json"))]
environment_path = out_dir / "environment.json"
manifest = {
    "benchmark": "megatron_matrix",
    "megatron_dir": str(megatron_dir),
    "megatron_commit": subprocess.check_output(
        ["git", "-C", str(megatron_dir), "rev-parse", "HEAD"], text=True
    ).strip(),
    "megatron_ref": megatron_ref,
    "image": megatron_image,
    "environment": json.loads(environment_path.read_text()) if environment_path.is_file() else None,
    "records": records,
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY

printf 'Megatron 矩阵结果: %s\n' "${OUT_DIR}/report.md"
