# MiniTrainBench

MiniTrainBench 是一个小型、可复现的分布式 GPT-like 训练 benchmark，
用于对比 PyTorch DDP 和 FSDP。项目使用合成 token 数据，因此不依赖数据集下载。

## 面向训练基础设施的能力展示

- 使用 DDP、FSDP、NCCL 和 Gloo 的 PyTorch distributed 启动与运行方式。
- 分析 DDP 吞吐优势与 FSDP 显存分片之间的实际取舍。
- 覆盖 all-reduce、all-gather、reduce-scatter 的通信 microbenchmark。
- 实现最小训练 Runtime：`TrainingConfig`、`TrainState`、`StepMetrics`、`Trainer`、
  deterministic synthetic data、distributed checkpoint/resume。
- 使用可插拔 strategy 抽象隔离 DDP/FSDP 包装逻辑，并支持 checkpoint retention。
- 通过 Docker 复现 GPU 实验，并通过非 GPU CI 做 smoke test。
- 自动生成包含扩展效率、显存节省、repeat 统计和 Runtime 状态的 Markdown 报告。

简历描述示例：

> 构建了一个 Docker 化的分布式 LLM 训练 benchmark，对比 PyTorch DDP/FSDP 在
> 1/2/4/8 卡下的吞吐、step time、显存和 NCCL collective 行为，并提供 CPU CI
> smoke test、可插拔训练策略、分布式 checkpoint/resume 和可复现 Markdown 报告。

## 训练框架能力点

- `TrainingStrategy` registry 将 DDP/FSDP 的模型包装、进程组需求和策略名称从
  `Trainer` 中拆出，贴近训练框架里的 strategy/plugin 设计。
- checkpoint 生命周期包含临时目录、DCP 保存、`metadata.json`、中文
  `metadata_zh.md`、`READY` 标记、`latest` 指针和恢复配置校验。
- `--keep-last` 支持只保留最近 N 个 READY checkpoint；pruning 在新 checkpoint
  完整发布后执行，避免删除唯一可恢复点。
- `--resume latest` 会优先读取 `latest` 指针；如果指针损坏或指向半成品目录，
  Runtime 会扫描 `step_*` 并跳过没有 `READY` 的目录。
- `scripts/run_runtime_resume_smoke.sh` 提供 preemption/resume 复现实验：先保存，
  再从 latest checkpoint 继续训练。

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

脚本会顺序运行 1/2/4/8 卡 DDP、1/2/4/8 卡 FSDP、8 卡 NCCL collective，
并重新生成 `results/report.md`。如需更长实验，可以通过 `GPUS`、`STEPS`、
`WARMUP_STEPS`、`REPEAT`、`OUT_DIR`、`COMM_NPROC` 或模型规模相关环境变量覆盖
默认配置。没有 8 卡时可使用 `GPUS="1 2 4" COMM_NPROC=4` 降级运行。

按 GPU 数运行短版 DDP benchmark：

```bash
mkdir -p results
for n in 1 2 4 8; do
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
for n in 1 2 4 8; do
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
  torchrun --standalone --nproc_per_node=8 -m minitrainbench comm \
  --device cuda --backend nccl \
  --sizes 1024,1048576,16777216 --warmup 10 --iters 50 \
  --output results/nccl_8gpu.json
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

验证训练 Runtime 的 checkpoint/resume：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench train \
  --strategy ddp --precision bf16 \
  --batch-size 2 --seq-length 256 \
  --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
  --steps 4 --warmup-steps 1 \
  --checkpoint-dir results/runtime_ckpt/ddp_2gpu --save-every 2 --keep-last 3 \
  --output results/ddp_2gpu_ckpt.json

docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench train \
  --strategy ddp --precision bf16 \
  --batch-size 2 --seq-length 256 \
  --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 \
  --steps 2 --warmup-steps 0 \
  --checkpoint-dir results/runtime_ckpt/ddp_2gpu --resume latest \
  --save-every 2 --keep-last 3 \
  --output results/ddp_2gpu_resume.json
```

运行默认 2 卡 FSDP/BF16 preemption/resume smoke：

```bash
IMAGE=minitrainbench:gpu scripts/run_runtime_resume_smoke.sh
```

如需切到 DDP 或调整保留策略：

```bash
STRATEGY=ddp KEEP_LAST=1 scripts/run_runtime_resume_smoke.sh
```

从保存的 JSON 结果生成 Markdown 报告：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m minitrainbench report \
  --input results/ddp_1gpu.json results/ddp_2gpu.json \
          results/ddp_4gpu.json results/ddp_8gpu.json \
          results/fsdp_1gpu.json results/fsdp_2gpu.json \
          results/fsdp_4gpu.json results/fsdp_8gpu.json \
          results/nccl_8gpu.json \
  --output results/report.md
```

## 实验表格

下表在当前主机生成，机器可用 8x NVIDIA A100-SXM4-40GB。实验使用本地
PyTorch 2.10.0 + CUDA 13.0 镜像构建的 `minitrainbench:gpu`。每行使用
23.2M 参数 GPT-like 模型、BF16、合成 token、单 rank batch size 2、
sequence length 256、2 个 warmup step 和 5 个测量 step。

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 | 12.37 | 2.15 | 34794.44 | 14.71 | 481.47 | 100.00% | - | - | 1 |
| ddp | 2 | bf16 | 0.06 | 15.40 | 2.34 | 56885.71 | 18.00 | 657.68 | 81.75% | - | - | 1 |
| ddp | 4 | bf16 | 0.09 | 17.78 | 2.87 | 100828.95 | 20.31 | 615.19 | 72.45% | - | - | 1 |
| ddp | 8 | bf16 | 0.06 | 15.71 | 2.36 | 225918.95 | 18.13 | 657.68 | 81.16% | - | - | 1 |
| fsdp | 1 | bf16 | 0.05 | 30.15 | 2.56 | 15571.04 | 32.88 | 479.77 | 100.00% | 0.35% | 18.17 | 1 |
| fsdp | 2 | bf16 | 0.05 | 33.41 | 1.33 | 29465.53 | 34.75 | 276.23 | 94.62% | 58.00% | 16.75 | 1 |
| fsdp | 4 | bf16 | 0.07 | 35.71 | 1.15 | 56058.97 | 36.53 | 208.09 | 90.01% | 66.17% | 16.22 | 1 |
| fsdp | 8 | bf16 | 0.06 | 32.29 | 0.92 | 124391.38 | 32.93 | 175.55 | 99.86% | 73.31% | 14.80 | 1 |

8 卡 NCCL collective 结果：

| 操作 | 元素数 | 延迟 (ms) | 带宽 (GB/s) |
| --- | ---: | ---: | ---: |
| all_reduce | 1024 | 0.052 | 0.080 |
| all_gather | 1024 | 0.185 | 0.177 |
| reduce_scatter | 1024 | 0.059 | 0.560 |
| all_reduce | 1048576 | 0.117 | 35.822 |
| all_gather | 1048576 | 0.347 | 96.739 |
| reduce_scatter | 1048576 | 0.245 | 137.122 |
| all_reduce | 16777216 | 0.724 | 92.734 |
| all_gather | 16777216 | 3.349 | 160.314 |
| reduce_scatter | 16777216 | 2.223 | 241.493 |

## 瓶颈分析

DDP 会在每个 rank 上保留完整的模型参数、梯度和优化器状态。它的主要分布式
开销来自梯度 all-reduce。这个 8x A100 full-node 短跑中，DDP 从 1 卡的
34.8k tokens/sec 扩展到 8 卡的 225.9k tokens/sec，扩展效率为 81.2%；最大显存
则在 481-658 MB 区间。forward/backward 加 optimizer 时间在 8 卡约为 18.1 ms，
说明当前 23.2M 小模型仍能在单节点内维持较高的扩展效率。

FSDP 会分片参数、梯度和优化器状态，因此可以降低稳定状态下的显存占用；代价是
在包裹的 Transformer block 周围引入参数 all-gather 和梯度 reduce-scatter
通信。在同一轮 8 卡实验中，FSDP 的最大显存从 1 卡的 479.8 MB 降到 175.5 MB，
相对 8 卡 DDP 节省 73.3%；吞吐从 15.6k 提升到 124.4k tokens/sec，step time 为
32.9 ms。它仍慢于 DDP，但相对 1 卡的扩展效率接近 100%，说明完整节点上的分片
收益能够摊薄部分通信成本。对更大模型，FSDP 的显存优势通常会更重要。

8 卡 NCCL 结果进一步解释了这个取舍：1024 元素的小 collective 仍受固定延迟限制；
16M 元素时 all-reduce、all-gather、reduce-scatter 分别达到 92.7、160.3、241.5
GB/s。FSDP 的 all-gather/reduce-scatter 在大 tensor 下有较高带宽，但每个 block
重复触发 collective，仍会给小模型带来可见的调度和同步开销。

可以结合通信 JSON 分析 collective 延迟、带宽和训练 step time 的关系。比较结果时
应保持模型、精度、local batch、sequence length、warmup 和测量 step 数一致。
activation checkpointing 通过额外重计算换取更低 activation 显存；gradient
accumulation 则通过在同步点之间累积更多计算，减少优化器更新频率。当前表格只跑了
`repeat=1`，适合展示 full-node 覆盖；用于严谨性能结论时应使用 `REPEAT=3` 或更高。

## 训练 Runtime 设计

`minitrainbench train` 现在由 `Trainer` 驱动。`TrainingConfig` 统一记录模型、
精度、batch、gradient accumulation、warmup、steps 和 seed；`TrainState`
记录 `global_step`、`micro_step`、`tokens_seen`、seed 与配置 fingerprint；
`StepMetrics` 拆分 data、forward/backward、optimizer 和整体 step time。

`TrainingStrategy` 是 `Trainer` 使用的策略插件接口。当前 registry 内置
`DDPStrategy` 和 `FSDPStrategy`，分别负责声明是否需要初始化进程组，以及如何
包装模型。这样后续扩展 ZeRO、设备后端或实验性 wrapper 时，不需要继续膨胀
`Trainer` 主循环。

synthetic token 数据按 `seed + global_step + rank` 的确定性规则生成。恢复训练时，
Runtime 从 checkpoint 中的 `global_step` 继续生成下一个 batch，避免重复消费或
跳过 synthetic step。

checkpoint 使用 `torch.distributed.checkpoint` 保存模型、optimizer 和训练状态。
DDP 与 FSDP 走同一套保存/加载入口，FSDP 可保留 sharded model/optimizer state。
保存目录采用 `step_00000010/` 形式；只有包含 `READY` 标记的目录会被视为可恢复。
写入过程先进入临时目录，所有 rank 完成 DCP 保存后再由 rank 0 写入 `metadata.json`、
中文 `metadata_zh.md`、READY 标记和 `latest` 指针，降低半成品 checkpoint 被误用的
风险。

当前 v1 只支持同 strategy、同 precision、同 world size、同模型配置和同关键训练
参数恢复；不匹配时会立即拒绝，并打印具体字段差异。跨 world size resharding、
异构后端迁移和 profiler trace 暂不放入这个最小 Runtime。

`--keep-last N` 用于控制 checkpoint retention。`N=3` 是默认值，`N=0` 表示保留
所有历史 checkpoint。清理逻辑只删除带 `READY` 的旧 checkpoint，不会把临时目录
误认为可恢复点。

## CPU CI

GitHub Actions 会安装 CPU 版 PyTorch wheel，并运行 tiny GPT forward/backward
测试、单进程训练 smoke test、checkpoint/resume、确定性 synthetic data、两进程
Gloo collective test、Markdown 报告渲染和 `ruff check .`。NCCL 和 GPU 相关
FSDP 性能实验保留为本地 Docker benchmark。
