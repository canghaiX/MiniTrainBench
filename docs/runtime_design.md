# MiniTrainBench 训练 Runtime 设计

本文档说明本项目为什么把 benchmark 循环拆成最小训练 Runtime，以及 checkpoint
恢复时保证哪些不变量。

## 核心对象

- `TrainingConfig`：保存模型结构、精度、batch、gradient accumulation、warmup、
  steps、seed 等参数，并生成稳定的 `config_fingerprint`。
- `TrainState`：保存 `global_step`、`micro_step`、`tokens_seen`、seed 和配置
  fingerprint，是 checkpoint/resume 的最小训练状态。
- `StepMetrics`：保存 data、forward/backward、optimizer 和整体 step time，
  便于定位瓶颈来自输入生成、计算通信还是优化器更新。
- `Trainer`：负责分布式初始化、模型构建、DDP/FSDP 包装、训练循环、指标聚合和
  checkpoint 生命周期。

## 数据确定性

benchmark 使用合成 token，避免数据集下载和 tokenizer 影响性能测量。每个 rank
按 `seed + global_step * 1000003 + rank * 97003` 初始化独立 `torch.Generator`，
再生成当前 step 的 batch。恢复训练时从 checkpoint 的 `global_step` 继续，因此
下一步数据可复现，也不会重复消费已经训练过的 synthetic step。

## Checkpoint 流程

checkpoint root 下保存多个 `step_00000010/` 目录。一次保存的流程如下：

1. rank 0 清理同 step 的临时目录，并等待所有 rank 同步。
2. 所有 rank 调用 `torch.distributed.checkpoint.save` 写入模型、optimizer 和
   tensor 化的训练状态。
3. 所有 rank 再次同步，rank 0 写入 `metadata.json`、中文 `metadata_zh.md`、
   `READY` 和 `latest` 指针。
4. 临时目录通过目录级替换变成最终 `step_*` 目录。

只有带 `READY` 标记的目录才会被 `--resume` 或 `--resume latest` 使用。
`metadata.json` 面向机器解析，记录 strategy、precision、world size、模型配置、
配置 fingerprint、tokens_seen 和生成时间；`metadata_zh.md` 面向人工排查恢复失败
原因。

## 恢复限制

v1 恢复要求 strategy、precision、world size、模型配置和关键训练参数一致。
这覆盖了训练 Runtime 的基础安全性，也避免把跨 world size resharding、异构设备
迁移等更重的能力混入当前项目。后续如果要继续深化，可以增加 resharding、
checkpoint retention policy 和 profiler trace。
