#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source scripts/lib/docker_provenance.sh
source scripts/lib/gpu_memory_sampler.sh

: "${MEGATRON_DIR:?请设置 MEGATRON_DIR=/path/to/Megatron-LM}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.18.2}"
MEGATRON_IMAGE="${MEGATRON_IMAGE:-minitrainbench:megatron}"
OUT_DIR="${OUT_DIR:-results/megatron_smoke}"
TP="${TP:-1}"
PP="${PP:-1}"
WORLD_SIZE="${WORLD_SIZE:-8}"
NAME="${NAME:-tp${TP}_pp${PP}}"
TRIAL_INDEX="${TRIAL_INDEX:-1}"
MEASURED_ITERS="${MEASURED_ITERS:-20}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_HEADS="${NUM_HEADS:-16}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-auto}"

if [[ "${OUT_DIR}" == /* || "/${OUT_DIR}/" == *"/../"* ]]; then
  echo "OUT_DIR 必须是当前仓库下的相对路径" >&2
  exit 2
fi
if [[ ! -f "${MEGATRON_DIR}/pretrain_gpt.py" ]]; then
  echo "MEGATRON_DIR 中未找到 pretrain_gpt.py: ${MEGATRON_DIR}" >&2
  exit 2
fi
if (( WORLD_SIZE % (TP * PP) != 0 )); then
  echo "WORLD_SIZE 必须能被 TP*PP 整除" >&2
  exit 2
fi
DP=$((WORLD_SIZE / TP / PP))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
  echo "GLOBAL_BATCH_SIZE 必须能被 MICRO_BATCH_SIZE*DP 整除" >&2
  exit 2
fi

minitrainbench_assert_source_clean
MEGATRON_COMMIT="$(git -C "${MEGATRON_DIR}" rev-parse HEAD)"
EXPECTED_COMMIT="$(git -C "${MEGATRON_DIR}" rev-parse "${MEGATRON_REF}^{commit}")"
if [[ "${MEGATRON_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Megatron HEAD 与 ${MEGATRON_REF} 不一致；脚本不会修改外部仓库" >&2
  exit 2
fi

revision="$(minitrainbench_repo_revision)"
image_id="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{.Id}}')"
build_revision="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
base_image="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}')"
core_version="$(docker image inspect "${MEGATRON_IMAGE}" --format '{{index .Config.Labels "io.minitrainbench.megatron-core.version"}}')"
if [[ "${build_revision}" != "${revision}" && "${ALLOW_UNVERIFIED_PROVENANCE:-0}" != "1" ]]; then
  echo "Megatron 镜像 build revision 与源码 HEAD 不一致，请重新构建镜像" >&2
  exit 2
fi
if [[ "${base_image}" == nvcr.io/nvidia/pytorch:26.01-py3@sha256:* ]]; then
  environment_profile="ngc_pytorch_26_01"
elif [[ "${base_image}" == pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:* && \
        "${ALLOW_PYTORCH_FALLBACK:-0}" == "1" ]]; then
  environment_profile="pytorch_2_10_official_fallback"
else
  echo "正式 Megatron 证据必须使用固定 NGC base；官方 PyTorch fallback 需显式启用" >&2
  exit 2
fi
if [[ "${SEQUENCE_PARALLEL}" == "auto" ]]; then
  if (( TP > 1 )) && [[ "${environment_profile}" == "ngc_pytorch_26_01" ]]; then
    sequence_parallel=1
  else
    sequence_parallel=0
  fi
elif [[ "${SEQUENCE_PARALLEL}" == "1" || "${SEQUENCE_PARALLEL}" == "0" ]]; then
  sequence_parallel="${SEQUENCE_PARALLEL}"
else
  echo "SEQUENCE_PARALLEL 必须为 auto、0 或 1" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/records" "${OUT_DIR}/tensorboard"
trial_name="${NAME}_trial$(printf '%02d' "${TRIAL_INDEX}")"
LOG_PATH="${OUT_DIR}/logs/${trial_name}.log"
MEMORY_PATH="${OUT_DIR}/logs/${trial_name}.memory.csv"
ENVIRONMENT_PATH="${OUT_DIR}/logs/${trial_name}.environment.json"
PROVENANCE_PATH="${OUT_DIR}/logs/${trial_name}.provenance.json"
RECORD_PATH="${OUT_DIR}/records/${trial_name}.json"
TRAIN_ITERS=$((WARMUP_ITERS + MEASURED_ITERS))

probe='import importlib.metadata,json,platform,subprocess,torch; print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":str(torch.backends.cudnn.version()),"nccl":".".join(map(str,torch.cuda.nccl.version())),"driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0].strip(),"gpu":torch.cuda.get_device_name(0),"gpu_count":torch.cuda.device_count(),"megatron_core":importlib.metadata.version("megatron-core")},sort_keys=True))'
docker run --rm --gpus all --ipc=host --network=host \
  -v "${MEGATRON_DIR}:/megatron:ro" -e PYTHONPATH=/megatron \
  -w /megatron "${MEGATRON_IMAGE}" python3 -c "${probe}" >"${ENVIRONMENT_PATH}"

megatron_args=(
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" /megatron/pretrain_gpt.py
  --mock-data --tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE}"
  --num-layers "${NUM_LAYERS}" --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "$((4 * HIDDEN_SIZE))" --num-attention-heads "${NUM_HEADS}"
  --seq-length "${SEQ_LENGTH}" --max-position-embeddings "${SEQ_LENGTH}"
  --micro-batch-size "${MICRO_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}" --lr-decay-iters "${TRAIN_ITERS}"
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 1
  --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95
  --bf16 --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --normalization RMSNorm --no-persist-layer-norm --no-gradient-accumulation-fusion
  --swiglu --disable-bias-linear
  --untie-embeddings-and-output-weights --position-embedding-type rope --no-rope-fusion
  --attention-dropout 0.0 --hidden-dropout 0.0 --no-masked-softmax-fusion
  --transformer-impl "${TRANSFORMER_IMPL}" --distributed-backend nccl
  --eval-iters 0 --eval-interval 100000 --log-interval 1 --log-throughput
  --tensorboard-dir "/workspace/${OUT_DIR}/tensorboard/${trial_name}"
  --log-timers-to-tensorboard --log-memory-to-tensorboard
)
if (( sequence_parallel == 1 )); then
  megatron_args+=(--sequence-parallel)
fi
command=(
  docker run --rm --gpus all --ipc=host --network=host
  -v "${PWD}:/workspace" -v "${MEGATRON_DIR}:/megatron"
  -e PYTHONPATH=/megatron -e CUDA_DEVICE_MAX_CONNECTIONS=1
  -w /megatron "${MEGATRON_IMAGE}" "${megatron_args[@]}"
)
printf -v public_command '%q ' "${command[@]}"
public_command="${public_command//${MEGATRON_DIR}/<MEGATRON_DIR>}"
public_command="${public_command//${PWD}/<REPO_DIR>}"

python3 - "${PROVENANCE_PATH}" "${revision}" "${image_id}" "${base_image}" \
  "${build_revision}" "${public_command% }" <<'PY'
import json
import sys
from pathlib import Path

path, revision, image_id, base_image, build_revision, command = sys.argv[1:]
payload = {
    "git_revision": revision, "git_dirty": False,
    "image_ref": "minitrainbench:megatron", "image_id": image_id,
    "base_image": base_image, "build_revision": build_revision,
    "command": command, "complete": True, "missing_fields": [],
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

config_json="$(python3 - "${NAME}" "${TRIAL_INDEX}" "${TP}" "${PP}" "${DP}" \
  "${WORLD_SIZE}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${SEQ_LENGTH}" \
  "${NUM_LAYERS}" "${HIDDEN_SIZE}" "${NUM_HEADS}" "${VOCAB_SIZE}" \
  "${MEASURED_ITERS}" "${WARMUP_ITERS}" "${MEGATRON_REF}" \
  "${MEGATRON_COMMIT}" "${core_version}" "${base_image}" "${TRANSFORMER_IMPL}" \
  "${environment_profile}" "${sequence_parallel}" <<'PY'
import json
import sys

(name, trial, tp, pp, dp, world, micro, glob, seq, layers, hidden, heads, vocab,
 measured, warmup, ref, commit, core, base, impl, profile, sequence_parallel) = sys.argv[1:]
print(json.dumps({
    "name": name, "trial_index": int(trial),
    "tp": int(tp), "pp": int(pp), "dp": int(dp), "world_size": int(world),
    "precision": "bf16", "micro_batch_size": int(micro),
    "global_batch_size": int(glob), "seq_length": int(seq),
    "measured_iters": int(measured), "warmup_iters": int(warmup),
    "model_config": {"num_layers": int(layers), "hidden_size": int(hidden),
                     "num_attention_heads": int(heads), "vocab_size": int(vocab),
                     "sequence_length": int(seq)},
    "batch_config": {"micro_batch_size": int(micro),
                     "global_batch_size": int(glob)},
    "megatron": {"ref": ref, "commit": commit, "core_version": core,
                 "ngc_base_image": base, "transformer_impl": impl,
                 "environment_profile": profile,
                 "distributed_optimizer": True,
                 "sequence_parallel": bool(int(sequence_parallel))},
}, separators=(",", ":")))
PY
)"

sampler_pid=""
minitrainbench_memory_sampler_start "${MEMORY_PATH}" sampler_pid
set +e
timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" "${command[@]}" \
  >"${LOG_PATH}" 2>&1
returncode=$?
set -e
minitrainbench_memory_sampler_stop "${sampler_pid}" "${MEMORY_PATH}"

PYTHONPATH=src python3 -m minitrainbench.evidence megatron-record \
  --config-json "${config_json}" --log "${LOG_PATH}" --returncode "${returncode}" \
  --command "${public_command% }" --environment "${ENVIRONMENT_PATH}" \
  --provenance "${PROVENANCE_PATH}" --memory-samples "${MEMORY_PATH}" \
  --output "${RECORD_PATH}"

printf '%s\n' "${RECORD_PATH}"
exit "${returncode}"
