# MiniTrainBench 学习路线与源码导读

本文把仓库整理成一套训练 Infra 学习材料。目标不是背命令，而是能够从一次训练启动、一个
optimizer step、一次 collective、一次 checkpoint 和一次故障恢复，讲清楚数据如何流动、
状态如何变化、通信在哪里发生，以及怎样证明结果可信。

## 学习目标

完成本文后，应能够回答：

1. DDP、FSDP 和 ZeRO 分别切分了什么，为什么显存和通信行为不同？
2. gradient accumulation 为什么可能减少 DDP 通信，却可能增加 FSDP 显存？
3. checkpoint 怎样避免多 rank 保存时发布半成品？resume 怎样做到精确一致？
4. 一个慢 step 如何用 profiler 判断是计算、通信、optimizer 还是 straggler？
5. MoE 为什么需要 all-to-all，TP/PP/SP 为什么不能简单当成 DDP 的变体？
6. rank crash、NaN、NCCL hang 和 provenance 错误分别怎样检测和处理？

## 源码阅读顺序

不要一开始从 CLI 入口一路读到底。按下面的顺序，每读完一个模块都记录“输入、输出、
状态、collective、失败方式”五项：

| 顺序 | 文件 | 重点问题 |
| --- | --- | --- |
| 1 | `src/minitrainbench/model.py` | GPT-like block、activation checkpoint 放在哪里 |
| 2 | `src/minitrainbench/data.py` | seed、global step、rank 如何决定 synthetic batch |
| 3 | `src/minitrainbench/strategy.py` | DDP/FSDP 的 strategy/plugin 边界 |
| 4 | `src/minitrainbench/runtime.py` | Trainer 如何推进 step、计时和状态 |
| 5 | `src/minitrainbench/scheduler.py` | benchmark warmup 与 LR warmup 的区别 |
| 6 | `src/minitrainbench/checkpoint.py` | DCP、临时目录、READY、latest、retention |
| 7 | `src/minitrainbench/verification.py` | 分片 state、RNG、scheduler digest 如何比较 |
| 8 | `src/minitrainbench/communication.py` | collective、all-to-all、split 语义 |
| 9 | `src/minitrainbench/profiler.py` | trace、key averages、rank 聚合 |
| 10 | `src/minitrainbench/tensor_parallel.py` | Column/Row Parallel 的 forward/backward |
| 11 | `src/minitrainbench/deepspeed_benchmark.py` | adapter 为什么与核心 Trainer 分开 |

## 七天路线

### 第一天：跑通最小训练

```bash
python -m pytest -q
python -m minitrainbench train --device cpu --backend gloo \
  --strategy ddp --precision fp32 --steps 2 --warmup-steps 1
```

如果在默认暴露 GPU 的容器中跑 CPU/Gloo suite，应显式隐藏 GPU，避免 PyTorch DCP 的
文件写入路径自动选择 CUDA stream：

```bash
docker run --rm -e CUDA_VISIBLE_DEVICES= -v "$PWD:/workspace" \
  -w /workspace minitrainbench:test python -m pytest -q
```

阅读 `model.py`、`data.py`、`runtime.py` 和 `tests/test_smoke.py`。要能解释一个 optimizer
step 包含多少 micro-batch、loss 为什么要除以 `grad_accum_steps`，以及 tokens/sec 使用
的是 local token 还是 global token。

### 第二天：DDP、FSDP 与 gradient accumulation

阅读 `strategy.py`，重点比较 `wrap_model()`、同步上下文和梯度范数接口：

```bash
cat results/gradient_sync/report.md
minitrainbench train --help
```

面试重点：DDP 是完整参数副本，反向阶段通常 all-reduce 梯度；FSDP 分片参数、梯度和
optimizer state，forward/backward 期间需要 all-gather 和 reduce-scatter。DDP `auto` 使用
`no_sync()` 延迟前几个 micro-batch；FSDP `auto` 默认每个 micro-batch 同步以控制显存峰值。

### 第三天：checkpoint 与精确恢复

阅读 `checkpoint.py`、`verification.py`，再查看：

```bash
cat results/runtime_stability/report.md
cat results/rank_crash/report.md
cat results/runtime_determinism/verification.json
```

画出发布顺序：

```text
各 rank 写 DCP/RNG -> barrier -> metadata -> READY -> latest -> prune
```

要能解释：为什么 `READY` 是发布提交标记；为什么只保存 model/optimizer 不足以精确恢复；
为什么 FSDP checkpoint verify 不能简单把所有 sharded tensor 拼到 rank 0。

### 第四天：scheduler、gradient health 与故障注入

阅读 `scheduler.py` 和 `fault_tolerance.py`，区分 benchmark warmup、LR warmup、non-finite
fail-fast 和 rank crash。不能让某个 rank 提前抛异常而其他 rank 继续 collective，否则很
容易变成死锁。

当前项目验证的是 `SIGKILL` 后 torchrun 退出和手工 exact resume，不是 TorchElastic 自动
重启。面试时要主动说清楚这个边界。

### 第五天：通信与 profiler

```bash
cat results/nccl_8gpu.json
cat results/moe_comm/report.md
cat results/profile_8gpu/report.md
cat results/profile_8gpu/profile_summary.md
```

分析顺序：先看 step time 和 tokens/sec 稳定性，再看 data、forward/backward、optimizer
拆分，再看 NCCL op 的 latency、bandwidth 和调用次数，最后比较 rank min/p50/max 判断
straggler。没有 trace 时只说“未确定”，不把理论 overlap 说成观察结果。

小 tensor 通常更容易 latency-bound，大 tensor 才更接近 bandwidth-bound；结论必须结合
tensor size、collective 类型和硬件，不要绝对化。

### 第六天：TP、PP、SP、MoE 与 Megatron

阅读 `docs/parallelism_notes.md`、`tensor_parallel.py`、`docs/moe_training_notes.md` 和
`docs/megatron_case_study.md`：

- Column Parallel Linear 切输出维，Row Parallel Linear 切输入维；
- PP 把层切到不同 stage，micro-batch 可以摊薄 pipeline bubble；
- SP 切分 sequence 维 activation，通常和 TP 的 collective 语义一起设计；
- MoE router 决定 token 去哪个 expert，expert parallel 的 dispatch/combine 需要 all-to-all；
- 本项目实现 toy correctness 和外部 Megatron runner，没有复刻完整生产级 PP runtime。

### 第七天：形成面试答案

用下面的顺序讲项目，控制在 90 秒：

1. 问题：比较分布式训练策略的吞吐、通信和显存；
2. 核心：把 benchmark 组织成可恢复、可验证的最小 Runtime；
3. 机制：strategy、deterministic data、DCP/READY、RNG、Profiler、collective；
4. 证据：8 卡训练、repeat=3、memory pressure、rank crash exact resume；
5. 边界：单节点、synthetic data、toy TP/SP、Megatron 正式 NGC matrix 尚未发布；
6. 取舍：不为填能力矩阵而提交没有独占环境或 provenance 的数字。

## 高频问题速查

| 问题 | 回答骨架 |
| --- | --- |
| DDP 和 FSDP 怎么选？ | 小模型/通信友好时 DDP 简单且可能更快；显存受限时 FSDP 用分片换通信 |
| ZeRO-3 和 FSDP 有什么关系？ | 都可分片 parameter/gradient/optimizer，但 engine 生命周期和 checkpoint 接口不同 |
| 为什么 FSDP 不默认 `no_sync()`？ | accumulation window 可能保留未分片梯度，减少通信但抬高显存峰值 |
| checkpoint 怎样避免半成品？ | 临时目录、多 rank barrier、READY 标记、latest 只指向 READY |
| 为什么 resume 要保存 RNG？ | dropout 等随机路径只靠 step 和 seed 不足以保证和连续训练一致 |
| all-reduce 和 all-to-all 差异？ | all-reduce 汇总梯度；all-to-all 按目的 rank 交换 token，受 split 不均影响 |
| profiler 看到 NCCL 就说明通信是瓶颈吗？ | 不够，还要看占比、调用次数、tensor size、step breakdown 和 rank spread |
| 为什么没有完整 Megatron？ | 内部做可验证 Runtime，外部固定版本读码和 runner，避免低质量重写 |
| 为什么没有宣称自动容错？ | 当前验证 SIGKILL 后手工 exact resume，不是 elastic 自动重启 |

## 学习记录模板

```text
模块：
入口文件：
输入与输出：
每个 rank 保存什么状态：
发生哪些 collective：
正常路径：
失败路径：
我做的实验：
观察到的事实：
尚未验证的假设：
面试一句话：
```

完整错误记录见[错误记录与诊断手册](error_log.md)，完整问答见[项目复盘与面试指南](interview_guide.md)。
