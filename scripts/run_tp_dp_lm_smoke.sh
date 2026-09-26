#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:-results/tp_dp_lm/tp2_dp2.json}

PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
  python3 -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --module minitrainbench tp lm-train \
  --device cpu --backend gloo --tp-size 2 --precision fp32 \
  --vocab-size 32 --d-model 8 --n-heads 2 --n-layers 2 --mlp-hidden-size 16 \
  --batch-size 1 --seq-length 3 --steps 1 --warmup-steps 0 \
  --output "${OUTPUT}"
