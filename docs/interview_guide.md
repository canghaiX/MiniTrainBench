# MiniTrainBench 项目复盘与训练 Infra 面试指南

本文档用于复盘 MiniTrainBench 从 benchmark 到最小训练 Runtime 的实现过程，并准备
训练框架、训练 Runtime、分布式训练和性能工程相关面试。

建议先阅读[学习路线与源码导读](learning_guide.md)，建立 data、state、collective、
checkpoint 和 failure 的完整链路；再阅读[错误记录与诊断手册](error_log.md)，按“现象、
诊断、根因、修复、验证、边界”复盘真实问题。本文负责把这些内容压缩成面试表达。

## 一分钟介绍

MiniTrainBench 是一个 Docker 化的 PyTorch 分布式训练 benchmark 和最小训练 Runtime。
它使用合成 token 驱动小型 GPT-like 模型，支持 DDP、FSDP、DeepSpeed ZeRO、BF16、
gradient accumulation、activation checkpointing、分布式 checkpoint/resume、
PyTorch Profiler trace、NCCL collective microbenchmark、all-to-all MoE 通信和 toy
tensor parallel correctness check。

项目不仅比较吞吐和显存，还实现了 strategy 抽象、checkpoint 原子发布、retention、
每 rank RNG 保存，以及连续训练与中断恢复之间的精确状态校验。当前已在 8x A100
单节点上完成 1/2/4/8 卡 DDP/FSDP 和 8 卡 NCCL 实测，并用 CPU/Gloo CI 覆盖主要
Runtime 契约。

故障恢复证据包含一次真实 worker `SIGKILL`：项目确认 launcher 失败时 READY checkpoint
未变化，再手工重启并对 model、optimizer、scheduler、TrainState 和各 rank RNG 做 exact
verify。面试时必须说 `manual restart`，不能说成实现了 TorchElastic 自动恢复。

新实验结果能追到源码 commit、官方 base digest、容器 image ID、完整命令和软件栈。旧 JSON
保持兼容，但报告会将其标记为 provenance 不完整，不能和锁定环境的新矩阵混成同一批结论。

## 架构与重点

| 子系统 | 核心内容 | 面试中应强调的价值 |
| --- | --- | --- |
| `Trainer` | 训练循环、计时、状态推进、指标聚合 | 把 benchmark 循环拆成可维护 Runtime |
| `TrainingStrategy` | DDP/FSDP 包装、默认同步策略、`no_sync()` 上下文 | 用插件边界隔离并行策略 |
| Synthetic data | 按 `seed + global_step + rank` 生成 token | 无数据下载依赖且恢复后不重复/跳过 batch |
| CheckpointManager | DCP、临时目录、READY、latest、retention | 防止半成品 checkpoint 被恢复 |
| RNG checkpoint | 每 rank CPU/CUDA RNG state | dropout 等随机路径可精确恢复 |
| `checkpoint verify` | 比较模型、optimizer、TrainState、RNG digest | 从“能恢复”升级到“可证明恢复正确” |
| Repeat summary | 独立 trial、`mean/std/min/max` | 从单次数字升级到可信实验方法 |
| Profiler | 每 rank Chrome trace、top ops、step breakdown | 展示性能定位思路 |
| 8 卡性能证据 | memory pressure、rank/collective profiler 诊断 | 解释规模变化下的显存通信取舍 |
| Megatron case study | parallel groups、TP/PP、distributed optimizer、checkpoint | 对照 toy Runtime 与生产框架边界 |
| DeepSpeed adapter | ZeRO-2/ZeRO-3 benchmark、统一 JSON | 与业界训练栈横向对照 |
| all-to-all | equal / uneven split、MoE token dispatch | 补齐 expert parallel 的核心通信语义 |
| Tensor Parallel | Column/Row Parallel Linear correctness | 展示 TP 切分和梯度聚合的理解 |
| Report | 吞吐、step time、显存、扩展效率、Runtime 状态 | 让实验结论可读、可复现 |
| CI | CPU PyTorch + Gloo smoke | 不依赖 GPU 也能守住核心契约 |

## 如何解释能力矩阵边界

README 顶部的能力矩阵要主动讲成“范围控制”，而不是“没来得及做”。当前项目聚焦
pretraining runtime 和单节点分布式 infra：DDP/FSDP/ZeRO、checkpoint/resume、
profiler、collective、MoE all-to-all 和 toy TP 都有可运行证据。Multi-node 没做，是因为
没有稳定多机资源时很难提交可信的 rdvz、hostfile、跨节点 NCCL 结果；RLHF/GRPO 没做，
是因为它们属于 post-training pipeline，和本项目的训练 Runtime/通信性能主线不同。

Megatron-LM 也采用同样边界：项目没有低质量复刻整个框架，而是自己实现可验证的
toy TP/SP 与 Runtime 契约，再固定上游版本阅读关键链路并准备外部 8 卡 TP/PP/DP
runner。`core_v0.18.2` 的五组 8 卡 TP/PP/DP topology 已完成 compatibility smoke，
但运行在官方 PyTorch fallback 且 GPU 非独占，因此不能说成 NGC 性能 benchmark。

90 秒表述：

> 我没有复刻完整 Megatron，而是把项目分成两层：MiniTrainBench 内部实现可验证的
> DDP/FSDP、checkpoint、Profiler、MoE all-to-all 和 toy TP/SP；外部读取 Megatron 的
> parallel groups、pipeline schedule、distributed optimizer 和 checkpoint 关键链路，
> 并用固定版本跑通五组 8 卡 TP/PP/DP compatibility smoke。由于 NGC repeat=3 和独占卡
> 条件还没满足，我不展示这批 smoke 的性能数字。这样既证明真实框架启动和并行拓扑，
> 也如实区分兼容性证据、正式性能证据和 toy Runtime 的边界。

面试时可以这样说：我把仓库边界写清楚，是为了让 reviewer 快速知道哪些能力已经实现并
benchmark，哪些属于后续扩展，而不是把不完整的多机或 RLHF demo 混进主项目。

快速投递材料优先使用 [英文项目一页摘要](project_one_pager.md)。如果面试官追问“实验中
真正踩过什么坑”，沿 [锁定环境 GPU 重跑复盘](postmortem_locked_gpu_rerun.md) 讲环境锁定、
adapter schema、Megatron fused kernel capability 和性能证据门禁，不要虚构 NCCL timeout。

## 真实遇到的问题与解决方式

### 1. Gradient accumulation 没有自动减少 DDP 通信

**问题**：最初的 gradient accumulation 虽然将多个 micro-batch 的 loss 除以累积步数，
但每次 `backward()` 仍会触发 DDP gradient all-reduce。这样计算虽然累积了，通信却没有
减少。

**解决**：新增 `--gradient-sync-mode {auto,every,last}`。DDP 的 `auto` 解析为 `last`，
前面的 micro-batch 用 `model.no_sync()` 包住完整 forward/backward，只在最后一个
micro-batch 同步梯度。

**关键点**：`no_sync()` 必须覆盖 forward 和 backward。只包 `backward()` 会破坏 DDP
对 forward 内部 reducer 状态的预期。

**实测**：2 卡 A100、23.2M 参数模型、BF16、`grad_accum_steps=4` 下，DDP `auto/last`
从 `68.00 ms` 降到 `59.64 ms`，吞吐从 `30.1k` 提升到 `34.3k tokens/sec`。

### 2. FSDP 的 `no_sync()` 是通信与显存的显式取舍

**问题**：不能把 DDP 的默认末步同步直接照搬给 FSDP。FSDP 在 `no_sync()` window 中可能
保留未分片梯度，降低通信的同时抬高峰值显存。

**解决**：FSDP 的 `auto` 固定解析为 `every`，优先显存安全；用户可以显式选择 `last`。

**实测**：同一组 2 卡实验中，FSDP `last` 的 step time 为 `123.82 ms`，低于
`auto/every` 的 `130.76 ms`。这个小模型的峰值显存都约为 `267 MB`，因此不能声称
“FSDP `last` 一定不增加显存”；更大模型或更长 accumulation window 才更容易暴露代价。

### 3. “global step 可恢复”不等于“训练可精确恢复”

**问题**：synthetic data 已经按 step 和 rank 确定性生成，但只保存 model、optimizer、
global step 仍不够。启用 dropout、随机 augmentation 或某些 activation checkpointing
路径后，恢复进程的 PyTorch RNG 状态与连续训练不同，后续参数会漂移。

**解决**：checkpoint v3 在临时目录中保存 scheduler、每个 rank 的 CPU RNG 和本地 CUDA RNG state；
所有 rank 完成 DCP 和 RNG 文件写入后，才写 `READY` 并更新 `latest`。

**证据**：2 卡 FSDP、BF16、dropout 0.1 下，对比连续训练与中断 resume，`checkpoint verify`
得到 `exact_match=true`。模型、optimizer、scheduler、TrainState 和每 rank RNG digest
全部一致。

### 4. checkpoint 发布需要避免“看起来存在但不可用”

**问题**：多 rank 保存时，某些 shard、metadata 或 optimizer state 可能已经写入，而另一个
rank 尚未完成。直接把目录当 checkpoint 使用，会让 preemption 后的恢复读到半成品。

**解决**：

1. rank 0 清理同 step 的临时目录。
2. 所有 rank 写 DCP state 和自己的 RNG 文件。
3. barrier 后由 rank 0 写 metadata、中文说明、`READY` 与 `latest`。
4. 只把带 `READY` 的 `step_*` 目录视为可恢复点。
5. `--keep-last` 只在新 checkpoint 已经 READY 且 latest 更新后清理旧点。

**面试表述**：READY 不是“文件写完”的装饰，而是 checkpoint 发布完成的提交标记；恢复路径
只信任已发布状态。

### 5. 新增配置字段会破坏旧 checkpoint 指纹

**问题**：加入 `gradient_sync_mode` 后，旧 v1 checkpoint 的 config fingerprint 不包含该
字段。如果直接用新 fingerprint 比较，旧 checkpoint 会在 RNG 降级逻辑之前被错误拒绝。

**解决**：为旧 schema 保留 legacy fingerprint 计算。当旧 metadata 缺少同步模式且用户
使用默认 `auto` 时，允许功能性恢复，并固定采用旧版 `every` 同步语义；同时标记
`resume_deterministic=false`，因为 v1 没有 RNG state。

**边界**：用户显式指定新的同步模式时会被拒绝，避免“恢复成功但通信语义悄悄改变”。

### 6. FSDP checkpoint 不能用普通 tensor 假设校验

**问题**：PyTorch FSDP 的 state dict 可能包含 `ShardedTensor` 或 `DTensor`。首次实现
digest 时，把 `ShardedTensor` 当普通 tensor 调用 `reshape()`，导致校验命令失败。

**解决**：digest 先识别分片对象，按本 rank local shard 计算 digest，再通过
`all_gather_object` 以 rank 顺序聚合。这样无需把完整模型 all-gather 到单卡，也能比较
两份分布式状态。

**面试表述**：验证分片 checkpoint 时，不能只比较 rank 0 文件，也不能为了比较而把所有
state 物化到单卡；应保持分片加载和分片比较。

### 7. MoE 的核心不是参数同步，而是 token dispatch

**问题**：如果只看 DDP/FSDP 的 all-reduce、all-gather、reduce-scatter，很容易误以为分布式
训练的通信问题都一样。但 MoE/expert parallel 的瓶颈往往是 `all_to_all`，因为 router
决定 token 去向，rank 间会出现不均匀的 token dispatch/combine。

**解决**：在 `comm` 里补 `all_to_all`，并区分 `equal` 和 `uneven` split。`equal` 适合看链路
上限，`uneven` 更贴近真实 MoE 负载不均。README 和笔记里明确说明，MoE 性能分析不能直接
拿 all-reduce 结果替代。

**面试表述**：expert parallel 的关键不是“又多了一种 collective”，而是必须同时考虑 router
负载均衡、capacity factor、buffer packing 和 all-to-all 延迟。

### 8. Tensor Parallel 要先讲切分语义，再讲实现细节

**问题**：很多人一提 TP 就只说“把模型切到多卡上”，但面试官通常会继续追问切在哪一维、
为什么是 ColumnParallelLinear / RowParallelLinear、以及 backward 时怎么聚合梯度。

**解决**：补一个 toy tensor parallel check，用单卡 reference 对比 Column/Row Parallel
Linear 的 forward、input grad、weight grad 和 bias grad。`ColumnParallelLinear` 按输出
维切分，`RowParallelLinear` 按输入维切分并在 partial output 上做聚合。

**面试表述**：TP 不只是“切一半参数”，而是要让切分后的激活流、梯度流和 collective 语义
都能闭环；PP 还会引入 bubble，SP 则进一步减少 activation 显存。

### 9. 性能指标必须按分布式最慢 rank 统计

**问题**：只记录 rank 0 的局部计时会低估同步训练的真实 step time，因为训练下一步受最慢
rank 限制。

**解决**：CUDA 计时前后显式 synchronize；step 指标跨 rank 取 max，loss 取 mean，
显存取 max，再计算全局 tokens/sec。

**边界**：当前短跑默认 `repeat=1`，适合展示覆盖度。性能结论应使用 `REPEAT=3` 或更多，
并报告均值、标准差和环境信息。

### 10. GPU 代码不能替代 CPU CI

**问题**：NCCL/FSDP GPU 验证成本高，且 GitHub Actions 默认没有 GPU。

**解决**：CI 使用 CPU PyTorch 和 Gloo，覆盖 tiny GPT forward/backward、单进程训练、
两进程 DDP gradient accumulation、checkpoint/resume、legacy checkpoint 降级、
checkpoint verify、Gloo collective 和报告渲染。GPU 脚本保留为 Docker 证据链。

### 11. repeat 不能只是同一训练状态上的连续窗口

**问题**：如果 `repeat=3` 只是同一个模型和 optimizer 连续训练 3 个测量窗口，得到的
方差会混入 loss 曲线、optimizer 状态和缓存变化，不是真正独立 trial。

**解决**：`repeat > 1` 时每个 trial 重新初始化模型、optimizer、TrainState 和 synthetic
iterator，并与 checkpoint/resume 参数互斥。报告用 `mean ± std` 展示主指标，JSON 保留
每个 trial 的 raw metrics。

**面试表述**：benchmark repeat 和训练恢复要拆开，这样性能统计和 Runtime 状态语义都更
干净。

### 12. Profiler 不应该污染主 benchmark

**问题**：Profiler 会改变 kernel 调度、内存采样和 Python 开销。如果在 `train`
benchmark 默认打开，会让 tokens/sec 失真。

**解决**：新增 `minitrainbench profile` 独立入口，复用同一训练 step，但单独导出每 rank
Chrome trace、top ops 和 step breakdown。主 benchmark 负责稳定数值，profile 负责解释
为什么慢。

**面试表述**：性能工程不能只报吞吐，还要能从 step breakdown 和 trace 中定位是 compute、
optimizer 还是 collective 等待。

### 13. DeepSpeed ZeRO 不直接塞进 `TrainingStrategy`

**问题**：DeepSpeed Engine 会接管 backward、step、gradient accumulation 和 checkpoint。
如果把它强行塞进 DDP/FSDP 的 `Trainer`，两套生命周期会混在一起。

**解决**：新增独立 `deepspeed` 子命令，跑 ZeRO-2/ZeRO-3 benchmark 并归一化到现有 JSON
schema。报告层可以横向比较 DDP/FSDP/ZeRO，核心 Runtime 仍保持 DDP/FSDP checkpoint
正确性。

**面试表述**：这是一个 adapter 边界取舍，不是“不会接入”；优先保持核心 Runtime 的状态
语义清晰。

## 核心实测结果

### 8x A100 单节点短跑

| 策略 | GPU 数 | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDP | 1 | 34794.44 | 14.71 | 481.47 | 100.00% |
| DDP | 8 | 225918.95 | 18.13 | 657.68 | 81.16% |
| FSDP | 1 | 15571.04 | 32.88 | 479.77 | 100.00% |
| FSDP | 8 | 124391.38 | 32.93 | 175.55 | 99.86% |

8 卡 FSDP 相对 8 卡 DDP 节省 `73.31%` 峰值显存，但在该 23.2M 小模型上仍慢于 DDP。
这说明 FSDP 的价值首先是模型可训练性和显存扩展，而不是小模型吞吐必然更高。

### 8 卡 NCCL collective

16M 元素下，all-reduce、all-gather、reduce-scatter 分别达到 `92.7`、`160.3`、
`241.5 GB/s`。小 tensor 更偏 latency-bound；大 tensor 更能反映链路带宽。

## 高频面试题与回答要点

### DDP 和 FSDP 的核心区别是什么？

DDP 每个 rank 保留完整参数、梯度和 optimizer state，backward 时以 all-reduce 同步梯度。
它实现简单、对中小模型吞吐通常更好，但显存随模型完整复制。

FSDP 将参数、梯度和 optimizer state 分片。forward 前后需要参数 all-gather，backward
阶段需要 reduce-scatter，因此显存更低、通信模式更复杂。模型尚能放入 DDP 时，FSDP 不
一定更快；模型接近显存边界时，FSDP 是可训练性的关键路径。

### 为什么 DDP gradient accumulation 要用 `no_sync()`？

在累积 N 个 micro-batch 时，只有第 N 次 backward 前需要得到全局同步梯度。前 N-1 次
若仍 all-reduce，会重复支付通信成本。`no_sync()` 将同步延迟到最后一次 backward，
同时 loss 除以 N 保持与等效大 batch 一致的梯度尺度。

### 为什么 FSDP 默认不使用末步同步？

FSDP 的 `no_sync()` 会让未分片梯度在 accumulation window 内保留，通信减少但峰值显存
可能明显上升。因此本项目把它设计成显式 `last` 模式，而不是默认行为。实际策略取决于
模型规模、显存余量、网络带宽和 accumulation 长度。

### 精确 resume 至少要保存哪些状态？

至少包括：模型参数、optimizer state、学习率调度器状态、
训练进度、数据迭代位置或可复现数据 seed、每 rank RNG state，以及混合精度 scaler state
（本项目 BF16 不使用 GradScaler）。如果这些状态有任意缺失，恢复可能能跑，但不保证与
连续训练等价。

### 为什么要有 READY、latest 和 retention？

READY 表示一次多 rank 保存已完整发布；latest 提供稳定的恢复入口；retention 控制存储
占用。清理必须发生在新 checkpoint 发布后，否则 preemption 恰好发生时可能删掉唯一的
可恢复点。

### 为什么 checkpoint verify 不直接比较文件 hash？

DCP 的 shard 文件布局、metadata 和创建时间可能不同，但两份 checkpoint 的训练状态仍可
等价。verify 应加载状态后比较模型、optimizer、TrainState 和 RNG 的值；对 FSDP 则按
rank 比较 local shard 并聚合 digest，避免把完整 state 汇集到单卡。

### all-reduce、all-gather、reduce-scatter 分别在训练中做什么？

- all-reduce：DDP 同步梯度。
- all-gather：FSDP 在计算前临时获取完整参数。
- reduce-scatter：FSDP 聚合梯度后把结果分片回各 rank。
- all-to-all：MoE expert parallel 中按 expert/rank 重新分发 token。

回答时应补充：collective 的实际代价由消息大小、调用频率、拓扑、rank 数和计算通信重叠
共同决定，不能只看单次峰值带宽。

### MoE 为什么绕不开 all-to-all？

MoE 的 router 会把不同 token 分配给不同 expert。如果 expert 按 rank 分布，一个 rank
本地 batch 中的 token 需要发送给多个远端 rank，同时也会接收别的 rank 发来的 token。
这就是 all-to-all。真实瓶颈还包括 token permutation、capacity overflow、load balance
和最慢 rank 等待，不能只用 DDP all-reduce 的带宽来估算。

### Tensor Parallel 的 Column/Row Linear 怎么切？

Column parallel 按输出维切权重，每个 rank 计算一段输出 shard，必要时再 concat。
Row parallel 按输入维切权重，每个 rank 计算 partial output，再通过 all-reduce 求和。
MLP 里常见 column split 后接 row split，这样中间 hidden 可以保持分片，减少不必要的
all-gather。

### Pipeline bubble 和 Sequence Parallel 分别解决什么？

Pipeline parallel 把不同 layer 放到不同 stage，但 micro-batch 进入和离开流水线时会有
空泡。micro-batch 越少，bubble 占比越高；1F1B schedule 可以降低空闲和 activation
驻留。Sequence Parallel 则通常和 TP 配合，把部分 activation 沿 sequence 维切分，降低
长序列训练的激活显存，但会带来额外 collective 和随机性管理要求。

### 如何解释 8 卡 DDP 不是 8 倍加速？

同步、collective、负载不均、kernel launch、optimizer、数据准备和小模型计算量都会形成
串行或不可完全并行的部分。当前 DDP 8 卡扩展效率是 `81.16%`，对 23.2M 小模型的单节点
短跑已经说明通信开销可见，但未完全主导。

### 如何判断 benchmark 结果是否可信？

保持模型、序列长度、local batch、精度、warmup、测量 step、GPU 频率和软件版本一致；
CUDA 计时要同步；多 rank 用 max step time；至少 repeat 多次报告均值和方差；区分训练
吞吐与纯 collective microbenchmark；避免混用不同批次运行的结果。

### 投 Seed 时怎么讲这个项目？

可以按这条线讲：

1. 我先做了一个最小训练 Runtime，把 DDP/FSDP、BF16、gradient accumulation、checkpoint/resume 和 PyTorch Profiler 串起来。
2. 再补通信层证据，把 all-reduce、all-gather、reduce-scatter、all-to-all 以及 ZeRO 的差异拆开看。
3. 然后加上故障恢复和正确性检查，让项目从 benchmark 变成能解释训练系统边界的最小框架。
4. 最后用 doctor、多机 torchrun 模板、toy TP/MoE demo 和 CI smoke，把工程化和可复现性补齐。
5. 固定 Megatron `core_v0.18.2`，在外部源码上跑通五组 8 卡 TP/PP/DP compatibility smoke，
   同时把非 NGC、非独占环境的性能字段标成无效。

面试追问时可以主动强调边界：

- 多机和真实 MoE 训练没有做成完整生产系统，但已经有诊断和通信证据链。
- 仓库内 TP/SP 只做 toy correctness；真实 PP 通过外部 Megatron compatibility smoke 验证，
  但 NGC repeat=3 性能仍待补，没有把整个 Megatron runtime 重写一遍。
- RLHF/GRPO、推理和编译器方向不在这次项目目标里。

这样的讲法比较像训练框架岗位，而不是单纯的 benchmark 简介。

## 面试前自检

- 能在 90 秒内讲清项目目标、架构和一条实测结论。
- 能解释 DDP `no_sync()` 为什么覆盖 forward/backward。
- 能解释 FSDP `no_sync()` 的显存风险，而不是只说“它更快”。
- 能画出 checkpoint 从临时目录到 READY/latest 的发布流程。
- 能列出精确 resume 需要的状态，并说明本项目对 scheduler 做了 v3 checkpoint，
  但没有引入 FP16 GradScaler，因为当前主线使用 BF16/FP32。
- 能解释为什么 verify 要处理 `ShardedTensor`/`DTensor`。
- 能说明 CPU/Gloo CI 覆盖什么、GPU 证据脚本覆盖什么。
- 能解释为什么 repeat trial 要独立初始化，为什么 profiler 入口与 benchmark 分离。
- 能解释 ZeRO-2、ZeRO-3 与 FSDP 的显存/通信对比，以及为什么 DeepSpeed adapter 独立。
- 能解释 MoE token dispatch 为什么是 all-to-all，以及 equal/uneven split 分别看什么。
- 能解释 toy Tensor Parallel 如何验证 Column/Row Parallel Linear 的 forward/backward。
- 能解释 Megatron 为什么要求 `CUDA_DEVICE_MAX_CONNECTIONS=1`，以及 local torch LayerNorm
  为什么不能冒充已启用 sequence parallel。
- 能区分 `execution_complete`、`provenance.complete` 和 `performance_valid`，说明五组
  Megatron smoke 成功不等于已有正式性能结论。
- 能诚实说明当前限制：小模型、synthetic data、单节点、toy TP 而非完整 Megatron、
  无 PP/SP Runtime、无跨 world size resharding；仓库已有短跑基线，严谨性能结论需要跑
  `run_a100_stability_matrix.sh`。

## 可直接用于简历或自我介绍的表述

> 实现 Docker 化的最小 PyTorch 分布式训练 Runtime，支持 DDP/FSDP/DeepSpeed ZeRO、
> NCCL collective、all-to-all MoE 通信、toy Tensor Parallel correctness、BF16、
> activation checkpointing 和 1/2/4/8 卡 A100 benchmark；
> 实现 strategy 抽象、独立 repeat 统计、PyTorch Profiler trace、READY-based 分布式
> checkpoint 发布、每 rank RNG 精确恢复和 checkpoint state verify，并通过 CPU/Gloo CI
> 覆盖 DDP accumulation、resume、profiler smoke 与分布式通信 smoke。
