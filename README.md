# MiniTrainBench

MiniTrainBench 是一个小型、可复现的分布式 GPT-like 训练 benchmark，
用于对比 PyTorch DDP 和 FSDP。项目使用合成 token 数据，因此不依赖数据集下载。

## 面向训练基础设施的能力展示

- 使用 DDP、FSDP、NCCL 和 Gloo 的 PyTorch distributed 启动与运行方式。
- 分析 DDP 吞吐优势与 FSDP 显存分片之间的实际取舍。
- 覆盖 all-reduce、all-gather、reduce-scatter 的通信 microbenchmark。
- 通过 Docker 复现 GPU 实验，并通过非 GPU CI 做 smoke test。
- 自动生成包含扩展效率、显存节省和 repeat 统计的 Markdown 报告。

简历描述示例：

> 构建了一个 Docker 化的分布式 LLM 训练 benchmark，对比 PyTorch DDP/FSDP 在
> 1/2/4 卡下的吞吐、step time、显存和 NCCL collective 行为，并提供 CPU CI
> smoke test 与可复现 Markdown 报告。

## 环境

项目不会把 PyTorch 安装到宿主机 Python 环境中。请使用 Docker 构建 GPU 镜像：

```bash
docker build -t minitrainbench:gpu .
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m pytest
```

默认基础镜像是 `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`。
如果需要使用内网镜像，可以覆盖 `BASE_IMAGE`：

```bash
docker build \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagscale/cuda12.8.1-torch2.7.1-python3.10-te2.9:20260209 \
  -t minitrainbench:gpu .
```

CPU 开发和 CI 使用 `requirements-cpu.txt`，然后以 editable 模式安装项目：

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e .
```

## 复现

运行完整 A100 实验矩阵：

```bash
IMAGE=minitrainbench:gpu REPEAT=1 scripts/run_a100_matrix.sh
```

脚本会顺序运行 1/2/4 卡 DDP、1/2/4 卡 FSDP、4 卡 NCCL collective，
并重新生成 `results/report.md`。如需更长实验，可以通过 `GPUS`、`STEPS`、
`WARMUP_STEPS`、`REPEAT`、`OUT_DIR` 或模型规模相关环境变量覆盖默认配置。

按 GPU 数运行短版 DDP benchmark：

```bash
mkdir -p results
for n in 1 2 4; do
  docker run --rm --gpus all --ipc=host --network=host \
    -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
    torchrun --standalone --nproc_per_node="$n" -m minitrainbench train \
      --strategy ddp --precision bf16 --batch-size 2 --seq-length 256 \
      --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
      --steps 5 --warmup-steps 2 --repeat 1 \
      --output "results/ddp_${n}gpu.json"
done
```

运行同口径 FSDP benchmark：

```bash
for n in 1 2 4; do
  docker run --rm --gpus all --ipc=host --network=host \
    -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
    torchrun --standalone --nproc_per_node="$n" -m minitrainbench train \
      --strategy fsdp --precision bf16 --batch-size 2 --seq-length 256 \
      --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
      --steps 5 --warmup-steps 2 --repeat 1 \
      --output "results/fsdp_${n}gpu.json"
done
```

运行 NCCL collective benchmark：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=4 -m minitrainbench comm \
  --device cuda --backend nccl \
  --sizes 1024,1048576,16777216 --warmup 10 --iters 50 \
  --output results/nccl_4gpu.json
```

验证 BF16、activation checkpointing 和 gradient accumulation 组合：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench train \
  --strategy fsdp --precision bf16 --activation-checkpointing \
  --grad-accum-steps 2 --batch-size 1 --seq-length 256 \
  --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
  --steps 5 --warmup-steps 2 --repeat 3 \
  --output results/fsdp_checkpoint_accum_2gpu.json
```

从保存的 JSON 结果生成 Markdown 报告：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m minitrainbench report \
  --input results/ddp_1gpu.json results/ddp_2gpu.json \
          results/ddp_4gpu.json results/fsdp_1gpu.json \
          results/fsdp_2gpu.json results/fsdp_4gpu.json \
          results/nccl_4gpu.json \
  --output results/report.md
```

## 实验表格

下表在当前主机生成，机器可用 8x NVIDIA A100-SXM4-40GB。实验使用本地
PyTorch 2.10.0 + CUDA 13.0 镜像构建的 `minitrainbench:gpu`。每行使用
23.2M 参数 GPT-like 模型、BF16、合成 token、单 rank batch size 2、
sequence length 256、2 个 warmup step 和 5 个测量 step。

| 策略 | GPU 数 | 精度 | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 30362.52 | 16.86 | 481.47 | 100.00% | - | - | 1 |
| ddp | 2 | bf16 | 38857.41 | 26.35 | 567.13 | 63.99% | - | - | 1 |
| ddp | 4 | bf16 | 85459.30 | 23.96 | 615.19 | 70.37% | - | - | 1 |
| fsdp | 1 | bf16 | 15478.27 | 33.08 | 479.77 | 100.00% | 0.35% | 16.22 | 1 |
| fsdp | 2 | bf16 | 29016.16 | 35.29 | 274.86 | 93.73% | 51.54% | 8.94 | 1 |
| fsdp | 4 | bf16 | 16230.08 | 126.19 | 209.60 | 26.21% | 65.93% | 102.22 | 1 |

4 卡 NCCL collective 结果：

| 操作 | 元素数 | 延迟 (ms) | 带宽 (GB/s) |
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

## 瓶颈分析

DDP 会在每个 rank 上保留完整的模型参数、梯度和优化器状态。它的主要分布式
开销来自梯度 all-reduce，因此在 1/2/4 卡实验中，显存从 481 MB 小幅上升到
615 MB，同时吞吐从 30.4k tokens/sec 扩展到 85.5k tokens/sec。由于当前模型
较小，2 卡结果受到额外同步开销影响，扩展效率不高。

FSDP 会分片参数、梯度和优化器状态，因此可以降低稳定状态下的显存占用；代价是
在包裹的 Transformer block 周围引入参数 all-gather 和梯度 reduce-scatter
通信。在这组短跑中，FSDP 将最大显存从 1 卡的 479.8 MB 降到 4 卡的
209.6 MB，但 4 卡 step time 增长到 126.2 ms，因为模型太小，无法摊平
每个 block 上 all-gather 和 reduce-scatter 的额外开销。因此这里更适合把
FSDP 理解为显存扩展路径，而不是小模型吞吐优化路径。

可以结合通信 JSON 分析 collective 延迟、带宽和训练 step time 的关系。比较结果时
应保持模型、精度、local batch、sequence length、warmup 和测量 step 数一致。
activation checkpointing 通过额外重计算换取更低 activation 显存；gradient
accumulation 则通过在同步点之间累积更多计算，减少优化器更新频率。

## CPU CI

GitHub Actions 会安装 CPU 版 PyTorch wheel，并运行 tiny GPT forward/backward
测试、单进程训练 smoke test 和两进程 Gloo collective test。NCCL 和 GPU 相关
FSDP 性能实验保留为本地 Docker benchmark。
