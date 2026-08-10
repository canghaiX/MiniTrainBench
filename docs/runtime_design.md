# MiniTrainBench 训练 Runtime 设计

本文档说明本项目为什么把 benchmark 循环拆成最小训练 Runtime，以及 checkpoint
恢复时保证哪些不变量。

## 核心对象

- `TrainingConfig`：保存模型结构、精度、batch、gradient accumulation、warmup、
  steps、seed 和 gradient sync mode 等参数，并生成稳定的 `config_fingerprint`。
- `TrainState`：保存 `global_step`、`micro_step`、`tokens_seen`、seed 和配置
  fingerprint，是 checkpoint/resume 的最小训练状态。
- `StepMetrics`：保存 data、forward/backward、optimizer 和整体 step time，
  便于定位瓶颈来自输入生成、计算通信还是优化器更新。
- `Trainer`：负责分布式初始化、模型构建、DDP/FSDP 包装、训练循环、指标聚合和
  checkpoint 生命周期。
- `TrainingStrategy`：隔离 DDP/FSDP 的进程组需求和模型包装逻辑，是后续扩展
  训练策略的插件边界，同时决定 gradient accumulation 的默认同步策略。

Profiler 和 DeepSpeed ZeRO 没有直接塞进 `Trainer` 主循环。`profile` 命令复用
`Trainer` 的单 step 执行路径，只在独立入口里打开 PyTorch Profiler；DeepSpeed ZeRO
使用独立 adapter 运行 benchmark，不复用 DCP checkpoint/resume。这样核心 Runtime
仍然聚焦 DDP/FSDP 的训练状态、同步策略和 checkpoint 正确性。

## Strategy 生命周期

`Trainer` 先通过 registry 创建 `TrainingStrategy`，再用
`strategy.requires_process_group()` 决定是否必须初始化进程组。模型构建完成后，
`Trainer` 调用 `strategy.wrap_model()` 得到 DDP 或 FSDP 包装后的模型，主训练循环
不再关心具体 strategy 分支。

当前内置 `DDPStrategy` 和 `FSDPStrategy`。这个边界的目的不是把项目做复杂，而是
让训练循环、状态推进、指标聚合和 checkpoint 生命周期保持稳定；后续增加实验性
wrapper 或异构设备策略时，只需要补新的 strategy 实现和 registry 项。

本项目当前没有把 DeepSpeed 作为 `TrainingStrategy` 注册。原因是 DeepSpeed Engine
本身接管 backward、step、gradient accumulation 和 checkpoint 语义；如果直接混入
`Trainer`，会让 DDP/FSDP 的 DCP checkpoint 生命周期与 DeepSpeed checkpoint 生命周期
交叉。当前选择是用 `minitrainbench deepspeed` 做横向 benchmark，对外暴露统一 JSON
结果，但内部保持独立。

## Repeat 实验协议

`--repeat N` 用于性能稳定性统计，不用于训练恢复。`N > 1` 时，每个 trial 都重新创建
模型、optimizer、`TrainState` 和 deterministic synthetic iterator，并从相同 seed
开始。报告中的 `summary` 记录 `mean/std/min/max`，Markdown 主表渲染为 `mean ± std`。

`repeat > 1` 与 `--checkpoint-dir`、`--save-every`、`--resume` 互斥。这样可以避免把
“独立 benchmark trial”和“preemption/resume 训练状态推进”混在同一次运行里。

## Gradient Accumulation 同步

`--gradient-sync-mode` 提供 `auto`、`every` 和 `last` 三种模式。`every` 保持每个
micro-batch 同步，`last` 使用 wrapper 的 `no_sync()` 包住完整 forward/backward，
只在最后一个 micro-batch 同步。

`auto` 由 strategy 解析：DDP 默认 `last`，避免一个 optimizer step 内重复 all-reduce；
FSDP 默认 `every`，避免未分片梯度在整个 accumulation window 内保留而放大峰值显存。
FSDP 使用 `last` 是显式的通信换显存选择。Runtime 会在 JSON 和 Markdown 报告中记录
请求模式、解析模式和每 step 的同步 micro-batch 数。

## 数据确定性

benchmark 使用合成 token，避免数据集下载和 tokenizer 影响性能测量。每个 rank
按 `seed + global_step * 1000003 + rank * 97003` 初始化独立 `torch.Generator`，
再生成当前 step 的 batch。恢复训练时从 checkpoint 的 `global_step` 继续，因此
下一步数据可复现，也不会重复消费已经训练过的 synthetic step。

## Checkpoint 流程

checkpoint root 下保存多个 `step_00000010/` 目录。一次保存的流程如下：

1. rank 0 清理同 step 的临时目录，并等待所有 rank 同步。
2. 所有 rank 调用 `torch.distributed.checkpoint.save` 写入模型、optimizer 和
   tensor 化的训练状态，并写入各自的 CPU/CUDA RNG state 文件。
3. 所有 rank 再次同步，rank 0 写入 `metadata.json`、中文 `metadata_zh.md`、
   `READY` 和 `latest` 指针。
4. 临时目录通过目录级替换变成最终 `step_*` 目录。

只有带 `READY` 标记的目录才会被 `--resume` 或 `--resume latest` 使用。
v2 的 `metadata.json` 还记录 RNG state 版本。恢复时每个 rank 加载自己的 RNG 文件，
因此 dropout 和 activation checkpointing 等随机路径会从 checkpoint 后精确延续。
`metadata_zh.md` 面向人工排查恢复失败原因。旧 checkpoint 没有 RNG 文件时仍可恢复，
但 Runtime 会标记 `resume_deterministic=false` 并给出中文警告。旧 metadata 同时不含
gradient sync mode；当用户使用默认 `auto` 恢复时，Runtime 固定采用旧版的 `every`
同步语义，避免 DDP 在恢复后悄悄改变通信行为。

## Retention 与 latest

`--keep-last N` 控制 checkpoint retention，默认保留最近 3 个 READY checkpoint，
`N=0` 表示不清理历史 checkpoint。清理发生在新 checkpoint 完成、`READY` 写入且
`latest` 指针更新之后，因此不会删除当前唯一可恢复点。

`--resume latest` 优先读取 `latest` 指针。如果指针缺失、损坏或指向没有 `READY`
的半成品目录，Runtime 会扫描 `step_*` 目录并选择最新 READY checkpoint。这个逻辑
模拟训练平台里常见的 preemption/resume 场景：只恢复发布完成的 checkpoint。

## 精确校验与恢复限制

`minitrainbench checkpoint verify` 在同 world size 下重建 checkpoint 对应的 strategy、
模型和 optimizer，比较两份 checkpoint 的模型、optimizer、TrainState 和每 rank RNG
state digest。任一项不一致会以非零状态退出，并将诊断写入 JSON。

v2 恢复和 verify 都要求 strategy、precision、world size、模型配置和关键训练参数一致。
这覆盖了训练 Runtime 的基础安全性，也避免把跨 world size resharding、异构设备
迁移等更重的能力混入当前项目。后续如果要继续深化，可以增加 resharding、
分层 retention policy、多机 checkpoint 发现和 DeepSpeed checkpoint 对齐。

## Profiler 入口

`minitrainbench profile` 使用 PyTorch Profiler 采集每 rank Chrome trace。普通
benchmark 不默认启用 profiler，因为 profiler 会改变 step time；项目把“产出稳定数值”
和“定位性能瓶颈”拆成两个入口。

Profiler 输出包含每 rank trace、rank summary、rank 0 聚合 JSON 和 Markdown。Markdown
中保留 data、forward/backward、optimizer step breakdown，以及 CUDA/CPU top ops 和
collective 线索。trace 文件通常较大，默认由 `.gitignore` 排除；仓库只需要提交摘要和
复现命令。
