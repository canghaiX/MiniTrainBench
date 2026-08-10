# MiniTrainBench

MiniTrainBench is a small, reproducible benchmark for distributed GPT-like
training with PyTorch DDP and FSDP. It uses synthetic token data, so runs do
not depend on a dataset download.

## Environment

The project does not install PyTorch into the host Python environment. Build
the GPU image with Docker:

```bash
docker build -t minitrainbench:gpu .
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m pytest
```

The default image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`.
Override it when using an internal mirror:

```bash
docker build \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagscale/cuda12.8.1-torch2.7.1-python3.10-te2.9:20260209 \
  -t minitrainbench:gpu .
```

For CPU development and CI, install `requirements-cpu.txt` and then the
editable project:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e .
```

## Reproduction

Run one short DDP benchmark per GPU count:

```bash
mkdir -p results
for n in 1 2 4; do
  docker run --rm --gpus all --ipc=host --network=host \
    -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
    torchrun --standalone --nproc_per_node="$n" -m minitrainbench train \
      --strategy ddp --precision bf16 --batch-size 2 --seq-length 256 \
      --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
      --steps 5 --warmup-steps 2 \
      --output "results/ddp_${n}gpu.json"
done
```

Run the matching FSDP benchmark:

```bash
for n in 1 2 4; do
  docker run --rm --gpus all --ipc=host --network=host \
    -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
    torchrun --standalone --nproc_per_node="$n" -m minitrainbench train \
      --strategy fsdp --precision bf16 --batch-size 2 --seq-length 256 \
      --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
      --steps 5 --warmup-steps 2 \
      --output "results/fsdp_${n}gpu.json"
done
```

Run NCCL collectives:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=4 -m minitrainbench comm \
  --device cuda --backend nccl \
  --sizes 1024,1048576,16777216 --warmup 10 --iters 50 \
  --output results/nccl_4gpu.json
```

Exercise BF16 with activation checkpointing and gradient accumulation:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench train \
  --strategy fsdp --precision bf16 --activation-checkpointing \
  --grad-accum-steps 2 --batch-size 1 --seq-length 256 \
  --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
  --steps 5 --warmup-steps 2 \
  --output results/fsdp_checkpoint_accum_2gpu.json
```

Generate a Markdown report from saved results:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m minitrainbench report \
  --input results/ddp_1gpu.json results/ddp_2gpu.json \
          results/ddp_4gpu.json results/fsdp_1gpu.json \
          results/fsdp_2gpu.json results/fsdp_4gpu.json \
          results/nccl_4gpu.json \
  --output results/report.md
```

## Experiment Table

The table below was generated on this host with 8x NVIDIA A100-SXM4-40GB
available, using Docker image `minitrainbench:gpu` built from the local
PyTorch 2.10.0 + CUDA 13.0 image. Each row uses a 23.2M-parameter GPT-like
model, BF16, synthetic tokens, per-rank batch size 2, sequence length 256,
2 warmup steps, and 5 measured steps.

| Strategy | GPUs | Precision | Tokens/sec | Step time (ms) | Max memory (MB) |
| --- | ---: | --- | ---: | ---: | ---: |
| ddp | 1 | bf16 | 30362.52 | 16.86 | 481.47 |
| ddp | 2 | bf16 | 38857.41 | 26.35 | 567.13 |
| ddp | 4 | bf16 | 85459.30 | 23.96 | 615.19 |
| fsdp | 1 | bf16 | 15478.27 | 33.08 | 479.77 |
| fsdp | 2 | bf16 | 29016.16 | 35.29 | 274.86 |
| fsdp | 4 | bf16 | 16230.08 | 126.19 | 209.60 |

4-GPU NCCL collective results:

| Operation | Elements | Latency (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| all_reduce | 1024 | 0.054 | 0.076 |
| all_gather | 1024 | 0.135 | 0.122 |
| reduce_scatter | 1024 | 0.061 | 0.269 |
| all_reduce | 1048576 | 0.102 | 41.172 |
| all_gather | 1048576 | 0.232 | 72.433 |
| reduce_scatter | 1048576 | 0.129 | 129.768 |
| all_reduce | 16777216 | 0.648 | 103.577 |
| all_gather | 16777216 | 6.092 | 44.062 |
| reduce_scatter | 16777216 | 1.818 | 147.652 |

## Bottleneck Analysis

DDP keeps a full copy of model parameters, gradients, and optimizer state on
each rank. Its main distributed cost is gradient all-reduce, so memory rises
slightly from 481 MB to 615 MB across the 1/2/4 GPU runs while throughput
scales from 30.4k to 85.5k tokens/sec. The 2-GPU row is limited by the extra
synchronization relative to this small model size.

FSDP shards parameters, gradients, and optimizer state. This lowers the
steady-state memory footprint, but introduces parameter all-gather and
gradient reduce-scatter traffic around wrapped Transformer blocks. In this
short run, FSDP reduces max memory from 479.8 MB on 1 GPU to 209.6 MB on
4 GPUs, but 4-GPU step time grows to 126.2 ms because the model is too small
to amortize per-block all-gather and reduce-scatter overhead. FSDP is therefore
best interpreted here as a memory-scaling path, not a small-model throughput
win.

Use the communication JSON to relate collective latency and bandwidth to the
training step. Compare runs with the same model, precision, local batch,
sequence length, warmup, and number of measured steps. Activation checkpointing
trades extra recompute for lower activation memory; gradient accumulation
trades fewer optimizer steps for more work between synchronization points.

## CPU CI

GitHub Actions installs the CPU PyTorch wheel and runs a tiny GPT forward/backward
test, a single-process training smoke test, and a two-process Gloo collective
test. NCCL and GPU-specific FSDP performance runs remain local Docker benchmarks.
