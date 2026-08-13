# MiniTrainBench

MiniTrainBench 是一个小型、可复现的分布式 GPT-like 训练 benchmark，
用于对比 PyTorch DDP 和 FSDP。项目使用合成 token 数据，因此不依赖数据集下载。

面试复盘和训练 Infra 高频问题见 [项目复盘与面试指南](docs/interview_guide.md)。
MoE/expert parallel 通信笔记见 [MoE 训练笔记](docs/moe_training_notes.md)，
Megatron-style TP/PP/SP 笔记见 [并行训练笔记](docs/parallelism_notes.md)。
Megatron-LM 的真实框架读码对照见 [Megatron 工程 Case Study](docs/megatron_case_study.md)，
8 卡性能定位方法见 [8 卡 Profiler Case Study](docs/profiler_case_study_8gpu.md)。
多机与 NCCL 诊断见 [多机与 NCCL 诊断笔记](docs/multinode_nccl_diagnostics.md)。

## 能力矩阵

| 方向 | 状态 | 仓库证据 |
| --- | --- | --- |
| DDP/FSDP | implemented + benchmarked | 1/2/4/8 卡训练结果、repeat=3 稳定性矩阵 |
| DeepSpeed ZeRO | benchmark adapter | ZeRO-2/ZeRO-3 与 DDP baseline 同表对比 |
| Checkpoint/resume | DDP/FSDP implemented | DCP、READY/latest、RNG state、`checkpoint verify` |
| Profiler | implemented + 2/8 卡实测 | PyTorch Profiler summary、跨 rank step/collective/straggler 分析 |
| Fault tolerance | smoke + resume evidence | 精确恢复、半成品 checkpoint 跳过、配置不匹配拒绝 |
| LR scheduler / gradient health | implemented | constant/cosine、全局 gradient norm、梯度裁剪、全 rank fail-fast |
| MoE | all-to-all microbenchmark + routing demo | equal/uneven split、MoE token dispatch 设计笔记 |
| Tensor Parallel | toy correctness check + notes | Column/Row Parallel Linear、TP MLP、Sequence Parallel |
| Megatron-LM | case study + external runner | 固定 ref 的外部源码 smoke/matrix；官方环境实测尚未产出 |
| Memory pressure | 8 卡 benchmarked | 23.2M/168.5M/731.1M/2.60B 的 DDP/FSDP/ZeRO 成功与 OOM 证据 |
| Multi-node | doctor + scripts | torchrun 多机模板、NCCL 诊断文档 |
| RLHF/GRPO | not implemented | 当前聚焦 pretraining runtime / distributed infra |

## 面向训练基础设施的能力展示

- 使用 DDP、FSDP、DeepSpeed ZeRO、NCCL 和 Gloo 的分布式训练启动与运行方式。
- 分析 DDP 吞吐优势、FSDP 显存分片和 ZeRO optimizer/parameter sharding 之间的实际取舍。
- 覆盖 all-reduce、all-gather、reduce-scatter、all-to-all 的通信 microbenchmark。
- 实现最小训练 Runtime：`TrainingConfig`、`TrainState`、`StepMetrics`、`Trainer`、
  deterministic synthetic data、distributed checkpoint/resume。
- 使用可插拔 strategy 抽象隔离 DDP/FSDP 包装逻辑，并支持 checkpoint retention。
- 提供 DDP/FSDP gradient accumulation 同步策略，并展示通信与显存取舍。
- 保存每个 rank 的 RNG 状态，支持带 dropout 的精确 checkpoint/resume 校验。
- 支持独立 repeat trial，报告 `mean ± std`，避免单次短跑被当成严谨性能结论。
- 支持 PyTorch Profiler，导出每 rank Chrome trace 与 rank 0 Markdown 摘要。
- 汇总 rank min/p50/max、collective time 和 straggler ratio；overlap 无 trace 证据时明确
  标为“未确定”。
- 提供 `minitrainbench doctor` 检查 GPU、NCCL、网卡和 rendezvous 连通性。
- 提供 `minitrainbench fault smoke`，把精确 resume、半成品 checkpoint 和配置不匹配这些常见故障边界写成可复现证据。
- 核心 Runtime 默认使用 constant LR 以保持 benchmark 基线兼容；可切换 cosine scheduler，记录
  LR、全局 gradient norm 和裁剪步数，并在 loss/gradient 非有限时让所有 rank 一致 fail-fast。
- 提供 toy tensor parallel correctness check，验证 Column/Row Parallel Linear
  与单卡 reference 的 forward/backward 一致性，并补了 toy MLP 与 sequence parallel demo。
- 提供 toy MoE routing demo，记录 top-1 dispatch、capacity、overflow 和 load imbalance。
- 提供显存压力矩阵和固定版本的 Megatron-LM 外部运行脚本，为后续真实框架结果与
  toy TP/SP 的对照保留可复现入口。
- 通过 Docker 复现 GPU 实验，并通过非 GPU CI 做 smoke test。
- 自动生成包含扩展效率、显存节省、repeat 统计和 Runtime 状态的 Markdown 报告。

简历描述示例：

> 构建了一个 Docker 化的分布式 LLM 训练 benchmark，对比 PyTorch DDP/FSDP/DeepSpeed
> ZeRO 在 1/2/4/8 卡下的吞吐、step time、显存和 NCCL collective 行为，补充
> all-to-all MoE 通信 benchmark、toy tensor parallel / sequence parallel correctness check、
> CPU CI smoke test、Profiler trace、可插拔训练策略、精确分布式 checkpoint/resume、
> 故障恢复 smoke、doctor 诊断和
> 可复现 Markdown 报告。

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
- `--gradient-sync-mode auto` 会让 DDP 只在 gradient accumulation 的最后一个
  micro-batch 同步梯度；FSDP 默认每个 micro-batch 同步以控制未分片梯度的显存峰值，
  也可显式切换到 `last`。
- checkpoint v3 在 `READY` 前保存 scheduler、每个 rank 的 CPU/CUDA RNG 状态；`checkpoint verify`
  可比较两份 DDP/FSDP checkpoint 的模型、optimizer、scheduler、TrainState 和 RNG digest。
- `minitrainbench profile` 使用 PyTorch Profiler 采集每 rank trace，保留 step 拆分和
  collective 线索，便于从吞吐数字追到具体性能瓶颈。
- DeepSpeed ZeRO 作为独立 adapter 接入 `minitrainbench deepspeed`，不接管现有
  DCP checkpoint/resume，避免两套 engine 生命周期混在一起。
- `minitrainbench comm --operations all_to_all` 用 equal/uneven split 模拟 MoE
  expert parallel 的 token dispatch/combine 通信路径。
- `minitrainbench tp check` 用 toy Column/Row Parallel Linear 展示 Megatron-style
  tensor parallel 的切分语义，不把 TP/PP/SP 强行并入当前 Runtime。
- `minitrainbench tp mlp` 和 `minitrainbench tp sequence` 分别补 toy MLP 与
  sequence parallel correctness demo，能把 Megatron-style 并行讲得更完整。
- `minitrainbench moe route` 提供 toy MoE routing / dispatch / combine 证据，和
  all-to-all benchmark 对上实际通信形态。
- `minitrainbench doctor` 检查 GPU、NCCL、网卡和 rendezvous 环境。
- `minitrainbench fault smoke` 把 resume、half checkpoint、config mismatch 和 NaN 等
  常见故障边界转成可复现 smoke。
- `scripts/run_runtime_stability_smoke.sh` 对比连续训练与中断恢复的 scheduler 精确状态，
  并实际注入 NaN 验证全 rank fail-fast 和 READY checkpoint 不推进。

## 环境

项目不会把 PyTorch 安装到宿主机 Python 环境中。请使用 Docker 构建 GPU 镜像：

```bash
docker build -t minitrainbench:gpu .
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  python -m pytest
```

如果要运行 DeepSpeed ZeRO-2/ZeRO-3 对比，请构建可选镜像：

```bash
export HTTPS_PROXY=http://your-proxy.example:80
docker build --target gpu-deepspeed \
  --secret id=https_proxy,env=HTTPS_PROXY \
  -t minitrainbench:deepspeed .
```

该 target 默认安装 `deepspeed==0.19.4`，并设置 `DS_BUILD_OPS=0`，避免在
benchmark 环境里编译 fused optimizer 扩展。默认 `docker build -t minitrainbench:gpu .`
仍然只生成基础 GPU benchmark 镜像。

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

运行短跑 A100 覆盖矩阵：

```bash
IMAGE=minitrainbench:gpu REPEAT=1 scripts/run_a100_matrix.sh
```

脚本会顺序运行 1/2/4/8 卡 DDP、1/2/4/8 卡 FSDP、8 卡 NCCL collective，
并重新生成 `results/report.md`。如需更长实验，可以通过 `GPUS`、`STEPS`、
`WARMUP_STEPS`、`REPEAT`、`OUT_DIR`、`COMM_NPROC` 或模型规模相关环境变量覆盖
默认配置。没有 8 卡时可使用 `GPUS="1 2 4" COMM_NPROC=4` 降级运行。

运行更适合写进实验结论的 `repeat=3` 稳定性矩阵：

```bash
IMAGE=minitrainbench:gpu scripts/run_a100_stability_matrix.sh
```

该脚本默认跑 1/2/4/8 卡 DDP、1/2/4/8 卡 FSDP 和 8 卡 NCCL，使用
`warmup_steps=5`、`steps=20`、`repeat=3`，结果写到
`results/stability_repeat3/`。每个 repeat 都会重新初始化模型、optimizer 和
deterministic synthetic iterator，报告中主指标渲染为 `mean ± std`。

本仓库提交的结果证据目录：

- `results/stability_repeat3/report.md`：DDP/FSDP 1/2/4/8 卡 repeat=3 稳定性矩阵。
- `results/zero_repeat3/report.md`：DDP baseline、DeepSpeed ZeRO-2、ZeRO-3 repeat=3 对比。
- `results/moe_comm/report.md`：2/4/8 卡 all-to-all equal/uneven 通信结果。
- `results/tensor_parallel/report.md`：toy Tensor Parallel correctness check。
- `results/profile/profile_summary.md`：DDP/FSDP PyTorch Profiler 摘要；原始 trace 被 `.gitignore` 排除。
- `results/memory_pressure/report.md`：多模型规模 DDP/FSDP/ZeRO 的成功、OOM 和失败证据。
- `results/profile_8gpu/profile_summary.md`：8 卡 DDP/FSDP profiler 跨 rank 摘要。

`results/megatron_smoke/report.md` 当前以 `not_run` 明确记录外部官方源码不可用，未填写
任何性能数字；因此能力矩阵不将 Megatron 标为 `benchmarked`。runner 只有在外部官方
源码、固定 ref 与容器环境校验通过后才会写入真实矩阵记录。

运行显存压力矩阵：

```bash
IMAGE=minitrainbench:gpu DEEPSPEED_IMAGE=minitrainbench:deepspeed \
  scripts/run_memory_pressure_matrix.sh
```

脚本默认使用 8 卡，依次运行 small、medium、large、stress 四档 DDP/FSDP/ZeRO-2/3。
单项 OOM 或失败不会中断其余实验，而是写入 `records/*.json` 和最终报告。可以通过
`TIERS`、`STRATEGIES`、`WORLD_SIZE`、`STEPS` 和 `TIMEOUT_SECONDS` 缩小矩阵。
脚本默认拒绝在 GPU 已有超过 1 GB 显存占用时启动，避免并发作业污染结果；只有明确
接受该风险时才设置 `ALLOW_BUSY_GPUS=1`。

运行 8 卡 DDP/FSDP Profiler：

```bash
IMAGE=minitrainbench:gpu NPROC=8 GRAD_ACCUM_STEPS=4 \
  OUT_DIR=results/profile_8gpu scripts/run_profiler_matrix.sh
```

摘要包含 top CPU/CUDA/NCCL op、step breakdown、rank min/p50/max、collective time 和
straggler ratio。`key_averages()` 无法证明计算通信 overlap，因此报告只会在 Chrome
trace 时间线可确认时下结论。

运行外部 Megatron-LM smoke / TP-PP-DP 矩阵：

```bash
MEGATRON_DIR=/path/to/Megatron-LM \
MEGATRON_REF=core_v0.18.2 \
MEGATRON_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 \
  scripts/run_megatron_tp_pp_matrix.sh
```

脚本要求外部仓库 HEAD 与固定 ref 一致，但不会切换或修改外部仓库。默认运行
TP/PP=`1/1`、`2/1`、`4/1`、`2/2`、`1/4`，使用 mock data，并保存完整命令、commit、
日志解析结果和失败原因。MiniTrainBench 不包含 Megatron 源码，也不宣称实现完整 PP。

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

运行 MoE expert parallel 风格的 all-to-all benchmark：

```bash
IMAGE=minitrainbench:gpu scripts/run_moe_comm_matrix.sh
```

脚本默认跑 2/4/8 卡，并分别测试 equal/uneven split。单独运行某个规模时：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench comm \
  --device cuda --backend nccl \
  --operations all_to_all --all-to-all-mode both \
  --sizes 1024,1048576 --warmup 10 --iters 50 \
  --output results/all_to_all_2gpu.json
```

运行 toy tensor parallel correctness check：

```bash
IMAGE=minitrainbench:gpu scripts/run_tensor_parallel_smoke.sh
```

CPU/Gloo 也可以验证相同的切分语义：

```bash
torchrun --standalone --nproc_per_node=2 -m minitrainbench tp check \
  --device cpu --backend gloo --batch-size 1 --seq-length 2 \
  --in-features 8 --out-features 8 --output results/tp_check_cpu.json
```

运行 toy MoE routing、TP MLP / Sequence Parallel 和故障恢复 smoke：

```bash
IMAGE=minitrainbench:gpu scripts/run_moe_routing_smoke.sh
IMAGE=minitrainbench:gpu scripts/run_tensor_parallel_smoke.sh
IMAGE=minitrainbench:gpu scripts/run_fault_tolerance_smoke.sh
```

检查单机或多机 rendezvous / NCCL 环境：

```bash
python -m minitrainbench doctor --skip-connectivity --output results/doctor.json
NODE_RANK=0 RDZV_ENDPOINT=10.0.0.1:29500 scripts/run_multinode_torchrun.sh
```

验证 BF16、activation checkpointing 和 gradient accumulation 组合：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench train \
  --strategy fsdp --precision bf16 --activation-checkpointing \
  --grad-accum-steps 2 --gradient-sync-mode auto --batch-size 1 --seq-length 256 \
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

运行 Runtime 稳定性闭环：连续 6 step、3+3 step resume、scheduler digest verify 和真实
NaN all-rank fail-fast 注入：

```bash
IMAGE=minitrainbench:gpu NPROC=2 \
  scripts/run_runtime_stability_smoke.sh
```

本机 A100 证据见 [`results/runtime_stability/report.md`](results/runtime_stability/report.md)：
2 卡 FSDP/BF16 的连续训练与中断恢复 `exact_match=true`，model、optimizer、scheduler、
TrainState 和 RNG 五类 digest 全部一致；NaN 注入后所有 rank 检测到，`global_step`、
scheduler 和 latest READY checkpoint 均不推进。

如需切到 DDP 或调整保留策略：

```bash
STRATEGY=ddp KEEP_LAST=1 scripts/run_runtime_resume_smoke.sh
```

运行 2 卡 gradient accumulation 同步策略对比：

```bash
IMAGE=minitrainbench:gpu scripts/run_gradient_sync_matrix.sh
```

脚本使用 `grad_accum_steps=4`，比较 DDP `auto/every` 与 FSDP `auto/last`，
输出 JSON 和 `results/gradient_sync/report.md`。可通过 `NPROC`、`STEPS`、
`WARMUP_STEPS`、`REPEAT`、`GRAD_ACCUM_STEPS` 和模型规模相关环境变量覆盖默认值。

运行 2 卡 DDP/FSDP PyTorch Profiler trace：

```bash
IMAGE=minitrainbench:gpu scripts/run_profiler_matrix.sh
```

每个 rank 会生成 `rank_00000.trace.json` 形式的 Chrome trace，并由 rank 0 汇总
`profile_summary.json` 和 `profile_summary.md`。原始 trace 默认不提交到 Git；
可在本地用 `chrome://tracing` 或 Perfetto 打开，结合 Markdown 摘要判断
forward/backward、optimizer 或 collective 是否是主要瓶颈。

运行 DeepSpeed ZeRO-2/ZeRO-3 对比矩阵：

```bash
docker build --target gpu-deepspeed -t minitrainbench:deepspeed .
IMAGE=minitrainbench:deepspeed scripts/run_zero_matrix.sh
```

该脚本默认跑 1/2/4/8 卡 DDP baseline、ZeRO-2 和 ZeRO-3，使用与稳定性矩阵一致的
`warmup_steps=5`、`steps=20`、`repeat=3`，结果写到 `results/zero_repeat3/`。
DeepSpeed adapter 只负责 benchmark，不接入当前 DCP checkpoint/resume。

验证带 dropout 的精确 FSDP resume：

```bash
IMAGE=minitrainbench:gpu scripts/run_runtime_determinism_smoke.sh
```

该脚本比较连续 3 step 与“2 step 保存 + resume 1 step”，并写入
`results/runtime_determinism/verification.json`。也可以手动比较两份同 world size
checkpoint：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace" -w /workspace minitrainbench:gpu \
  torchrun --standalone --nproc_per_node=2 -m minitrainbench checkpoint verify \
  --device cuda \
  --left /workspace/results/runtime_determinism/continuous_fsdp_2proc/step_00000003 \
  --right /workspace/results/runtime_determinism/interrupted_fsdp_2proc/step_00000003 \
  --output /workspace/results/runtime_determinism/verification.json
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

## 实验方法

MiniTrainBench 区分两种实验口径：

- 短跑覆盖矩阵：默认 `warmup_steps=2`、`steps=5`、`repeat=1`，用于快速确认
  1/2/4/8 卡 DDP/FSDP/NCCL 都能跑通。
- 稳定性矩阵：默认 `warmup_steps=5`、`steps=20`、`repeat=3`，用于报告更可信的
  `mean ± std`。每个 repeat 都会独立重建模型、optimizer、训练状态和 synthetic data
  iterator，因此方差反映运行时波动，而不是同一训练状态连续推进后的混合窗口。

`repeat > 1` 与 checkpoint/resume 互斥。这样 benchmark trial 与训练 Runtime 恢复语义
保持分离，避免把“性能稳定性实验”和“中断恢复实验”混在同一份 JSON 里。

## 实验表格

下表是当前仓库已保存的 repeat=1 短跑基线，在当前主机生成，机器可用
8x NVIDIA A100-SXM4-40GB。实验使用本地
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

2 卡 gradient accumulation 同步策略实测：

| 策略 | 请求模式 | 实际模式 | 同步 micro-batch/step | Tokens/sec | Step time (ms) | 最大显存 (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ddp | auto | last | 1 | 34339.65 | 59.64 | 656.15 |
| ddp | every | every | 4 | 30118.36 | 68.00 | 655.06 |
| fsdp | auto | every | 4 | 15661.75 | 130.76 | 267.21 |
| fsdp | last | last | 1 | 16540.01 | 123.82 | 267.21 |

精确恢复校验使用 2 卡 FSDP、BF16、dropout 0.1、小型 17.4K 参数模型。连续 3 step
和“2 step 保存 + resume 1 step”的 `checkpoint verify` 结果为 `exact_match=true`：
模型、optimizer、TrainState 和每 rank RNG state digest 均一致。

## MoE 通信路径与 Megatron-style 并行

MoE expert parallel 的核心通信是 token dispatch/combine。每个 rank 持有部分 expert，
router 选择 top-k expert 后，token 需要按目标 expert 所在 rank 重新打包并通过
all-to-all 发送。`equal` split 可以观察理想均衡下的带宽，`uneven` split 更接近真实
router 产生的负载不均，也更容易暴露 straggler 和 buffer shape 压力。设计细节见
[MoE 训练笔记](docs/moe_training_notes.md)。

Tensor Parallel 解决的是单层矩阵乘法如何跨 rank 切分。`ColumnParallelLinear` 按输出维
切权重，`RowParallelLinear` 按输入维切权重并在 partial output 上做 all-reduce。
`minitrainbench tp check` 不追求完整 Megatron 训练，而是用小矩阵验证 forward/backward
与单卡 reference 一致，作为 TP/PP/SP 面试讨论的代码证据。更多笔记见
[并行训练笔记](docs/parallelism_notes.md)。

## 性能定位证据

`minitrainbench profile` 用 PyTorch Profiler 在每个 rank 采集 trace。报告中的
step breakdown 先回答“慢在哪一段”：data、forward/backward 还是 optimizer；
rank top ops 再帮助定位到 attention、matmul、optimizer 或 NCCL collective。Chrome
trace 适合继续检查 kernel 时间线、rank 间等待、collective 调用频率和计算通信重叠。

普通 benchmark 不默认开启 profiler，因为 profiler 会引入额外开销，影响 tokens/sec。
因此项目把“计时用 benchmark”和“定位用 profile”拆成两个入口：前者沉淀稳定数值，
后者沉淀性能证据。

### 8 卡 Profiler Case Study

同一 23.2M 模型、BF16、`grad_accum_steps=4` 下，DDP 平均 step time 为 99.11 ms，
FSDP 为 199.49 ms；DDP 每 rank 峰值显存约 569.65 MB，FSDP 为 131.83-134.30 MB。
DDP 主要 collective 是 all-reduce，FSDP 则在 profile window 内出现更高频的
all-gather/reduce-scatter。两组 rank `max/p50` 均低于 1.002，没有观察到明显
step-time straggler。collective event duration 可能与计算重叠，因此不能当作纯阻塞
时间；trace 未提交，overlap 结论保持“未确定”。完整分析见
[8 卡 Profiler Case Study](docs/profiler_case_study_8gpu.md)。

## 显存压力矩阵

8 卡 BF16、activation checkpointing、短测量窗口下，模型从 23.2M 扩大到 2.60B 参数：

| 规模 | DDP | FSDP | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 23.2M | 133.6K tok/s, 568 MB | 91.0K, 123 MB | 114.6K, 2057 MB | 25.2K, 1118 MB |
| 168.5M | 56.7K, 3892 MB | 48.8K, 501 MB | 62.9K, 2828 MB | 10.3K, 1732 MB |
| 731.1M | 18.8K, 16904 MB | 23.6K, 1844 MB | 31.6K, 5472 MB | 4.5K, 3974 MB |
| 2.60B | OOM | 16.6K, 6371 MB | 18.6K, 12582 MB | 4.3K, 10153 MB |

这组结果给出的边界比小模型 baseline 更明确：DDP 在 small/medium 仍有较低框架开销，
但显存随参数规模快速上升；到 731.1M 时 FSDP 已同时取得更低显存和更高吞吐，到
2.60B 时 DDP OOM，而三种分片策略仍可训练。当前 ZeRO-3 在各档位均受细粒度参数
gather 和 engine 开销影响，不能把“分片更彻底”直接等价为“吞吐更高”。这些数值只有
3 measured steps，适合说明可训练边界，不替代 repeat=3 的稳定吞吐结论。完整命令、
状态和失败类型见 [显存压力报告](results/memory_pressure/report.md)。

## Megatron-LM Case Study

项目没有复刻完整 Megatron：内部 Runtime 负责可验证的 DDP/FSDP、checkpoint、Profiler、
MoE all-to-all 和 toy TP/SP；[Megatron 工程 Case Study](docs/megatron_case_study.md)
对照真实框架的 parallel groups、TP layers、pipeline schedule、distributed optimizer 和
distributed checkpoint。外部 runner 固定 `core_v0.18.2`，要求用户提供官方源码并记录
commit、容器、软件版本和完整命令。当前未在匹配的官方 Megatron 环境产出实测，因此
README 不展示 TP/PP 性能数字，也不把 Megatron 标为 benchmarked。

没有实现完整 Megatron、多机或 RLHF 是主动控制范围：本项目优先证明 pretraining runtime
的通信、显存、恢复和性能诊断能力。生产级 PP schedule、跨节点 fabric 验证和训练后阶段
需要独立的系统边界，不用未经验证的 toy 实现填充能力矩阵。

## ZeRO 对比边界

DeepSpeed ZeRO-2/ZeRO-3 通过 `minitrainbench deepspeed` 单独接入。ZeRO-2 主要分片
optimizer state 和 gradients；ZeRO-3 进一步分片 parameters，因此更接近 FSDP 的显存
优化路径。报告会把 `deepspeed_zero2`、`deepspeed_zero3` 与同 world size 的 DDP
baseline 对比显存和 step time。

DeepSpeed adapter 不复用当前 DDP/FSDP 的 DCP checkpoint/resume，因为 DeepSpeed Engine
有自己的状态管理和 checkpoint 生命周期。当前项目选择把 ZeRO 作为横向 benchmark，
而不是把两套 checkpoint 语义混在同一个 `Trainer` 里。

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

MoE 的 all-to-all 不能只看平均带宽。equal split 主要反映均衡 token dispatch 的链路
能力；uneven split 还会受到 token 数量不均、不同 rank 的 buffer 大小和最慢 rank 的
影响。实际 MoE runtime 需要同时观察 router load balance、capacity overflow 和
all-to-all latency，不能用 all-reduce 的结果直接替代。

可以结合通信 JSON 分析 collective 延迟、带宽和训练 step time 的关系。比较结果时
应保持模型、精度、local batch、sequence length、warmup 和测量 step 数一致。
activation checkpointing 通过额外重计算换取更低 activation 显存；gradient
accumulation 则通过在同步点之间累积更多计算，减少优化器更新频率。实现上，DDP 若在
每个 micro-batch 同步，会重复触发梯度 all-reduce；本轮 2 卡实验中，`auto` 解析为
末步同步后将 step time 从 68.00 ms 降到 59.64 ms。FSDP 默认保持每 micro-batch
同步，避免未分片梯度在 accumulation window 内累积；显式 `last` 在本轮短跑将 step
time 从 130.76 ms 降到 123.82 ms，但这个 23.2M 模型没有观察到额外峰值显存，不能将
该现象外推到更大模型。当前表格只跑了 `repeat=1`，适合展示 full-node 覆盖；用于严谨
性能结论时应使用 `REPEAT=3` 或更高。

## 训练 Runtime 设计

`minitrainbench train` 现在由 `Trainer` 驱动。`TrainingConfig` 统一记录模型、
精度、batch、gradient accumulation、warmup、steps 和 seed；`TrainState`
记录 `global_step`、`micro_step`、`tokens_seen`、seed 与配置 fingerprint；
`StepMetrics` 拆分 data、forward/backward、optimizer 和整体 step time。

`TrainingStrategy` 是 `Trainer` 使用的策略插件接口。当前 registry 内置
`DDPStrategy` 和 `FSDPStrategy`，分别负责声明是否需要初始化进程组，以及如何
包装模型。Profiler 复用 `Trainer` 的 step 执行能力，但通过独立 `profile` 命令采集
trace；DeepSpeed ZeRO 通过独立 adapter 进入 benchmark，不接管 `Trainer` 的
checkpoint 生命周期。

synthetic token 数据按 `seed + global_step + rank` 的确定性规则生成。恢复训练时，
Runtime 从 checkpoint 中的 `global_step` 继续生成下一个 batch，避免重复消费或
跳过 synthetic step。

checkpoint 使用 `torch.distributed.checkpoint` 保存模型、optimizer、scheduler 和训练状态。
DDP 与 FSDP 走同一套保存/加载入口，FSDP 可保留 sharded model/optimizer state。
保存目录采用 `step_00000010/` 形式；只有包含 `READY` 标记的目录会被视为可恢复。
写入过程先进入临时目录，所有 rank 完成 DCP 保存后再由 rank 0 写入 `metadata.json`、
中文 `metadata_zh.md`、READY 标记和 `latest` 指针，降低半成品 checkpoint 被误用的
风险。v3 checkpoint 还会在发布前写入 scheduler 状态和每个 rank 的 CPU/CUDA RNG 状态，
使带 dropout 或 activation checkpointing 的随机训练路径可以精确恢复。

当前 v3 只支持同 strategy、同 precision、同 world size、同模型配置和同关键训练
参数恢复；不匹配时会立即拒绝，并打印具体字段差异。跨 world size resharding、
异构后端迁移和 DeepSpeed checkpoint 接管暂不放入这个最小 Runtime。

旧版 v1/v2 checkpoint 缺少 scheduler 字段时，默认 constant scheduler 可以依据
`global_step` 重建；缺少 RNG 的旧 checkpoint 仍可功能性恢复，但会标记为非精确恢复。

`minitrainbench checkpoint verify` 会以保存时的 strategy 和模型配置重新构建训练
状态，在相同 world size 下加载两份 checkpoint，再对模型、optimizer、scheduler、
TrainState 和每 rank RNG state 做分布式 digest 比较。任一项不一致会写出诊断 JSON 并以
非零状态退出。

`--keep-last N` 用于控制 checkpoint retention。`N=3` 是默认值，`N=0` 表示保留
所有历史 checkpoint。清理逻辑只删除带 `READY` 的旧 checkpoint，不会把临时目录
误认为可恢复点。

## CPU CI

GitHub Actions 会安装 CPU 版 PyTorch wheel，并运行 tiny GPT forward/backward
测试、单进程训练 smoke test、checkpoint/resume、确定性 synthetic data、两进程
Gloo collective test、dropout 下的 exact checkpoint verify、Markdown 报告渲染和
`ruff check .`。本轮还覆盖 CPU PyTorch Profiler smoke、独立 repeat 语义、
DeepSpeed ZeRO config builder、all-to-all graceful skip 和 toy tensor parallel
correctness；NCCL、DeepSpeed GPU、MoE all-to-all 和 FSDP 性能实验保留为本地
Docker benchmark。
