# Megatron-LM 工程读码 Case Study

本文不是 Megatron 概念摘录，而是沿训练主链路对照 Megatron-LM 与 MiniTrainBench。
读码基线固定为 `core_v0.18.2`；上游版本升级时，应重新核对符号和行为，不能把本文当作
永久稳定的 API 文档。

## 1. Parallel Groups

### Megatron-LM 如何实现

`megatron/core/parallel_state.py` 负责把 global ranks 映射到 TP、PP、DP、CP、EP 等
process group。核心不是分别调用几次 `new_group()`，而是先确定维度顺序和 rank
generator，再为不同通信语义生成正交或组合 group。一个 rank 会同时属于多个 group：

- TP group 负责层内张量切分后的聚合。
- PP group 负责相邻 stage 的 activation/gradient P2P。
- DP group 负责副本间梯度同步或 distributed optimizer shard。
- EP group 负责 MoE expert 所有权和 token dispatch。

配置必须满足 `world_size = TP * PP * CP * DP`；引入 EP 后还需区分普通 DP 与 expert
data parallel group。group 初始化顺序必须在所有 rank 上一致，否则容易在启动阶段 hang。

### MiniTrainBench 当前如何实现

当前 `DistributedContext` 只维护一个默认 world process group。DDP/FSDP、collective、toy
TP 和 MoE demo 都直接复用这个 group，因此适合验证单一通信语义，但不能表达 TP=2、
PP=2、DP=2 这种正交拓扑。

### 差距和边界

MiniTrainBench 缺少 topology/rank generator、组合 process group 和 group 生命周期管理。
如果继续实现完整 TP/PP，这应当先于模型切分完成；否则不同 demo 会各自硬编码 rank
关系。本轮通过外部 Megatron matrix 验证真实 group 拓扑，不把一个残缺的 group manager
塞入现有 Runtime。

## 2. Tensor Parallel

### Megatron-LM 如何实现

`megatron/core/tensor_parallel/` 包含 Column/Row Parallel Linear、vocab parallel、
cross entropy、随机数和通信映射：

- Column Parallel Linear 按输出维切权重，QKV projection 和 MLP 第一层常采用该方式。
- Row Parallel Linear 按输入维切权重，对 partial output 做 reduce，attention output 和
  MLP 第二层常采用该方式。
- Vocab Parallel Embedding/LM head 按 vocabulary 维切分，并配合 vocab-parallel cross
  entropy，避免每个 rank 保存和物化完整 logits。
- sequence parallel 把 LayerNorm、Dropout 等 activation 沿 sequence 维切分，常与 TP 的
  reduce-scatter/all-gather 配套。

TP 不只是两个 Linear 类。真实实现还需处理参数初始化、RNG tracker、autograd collective、
异步通信、通信/计算 overlap 和 checkpoint sharded state。

### MiniTrainBench 当前如何实现

`tp check` 验证 Column/Row Parallel Linear 与单卡 reference 的 forward/backward；`tp mlp`
验证两者组成 MLP；`tp sequence` 验证 LayerNorm/Dropout 的 sequence shard 语义。这些测试
用于证明公式与梯度聚合正确，不接管主训练 Runtime。

### 差距和边界

当前没有 QKV/vocab parallel、TP RNG tracker、异步 overlap、完整 Transformer block 或
TP checkpoint。toy demo 的价值是 correctness，不应宣称具备 Megatron TP 训练能力。
外部 Megatron 的 TP=2/4 实验用于观察真实层内通信和每 rank 显存变化。

## 3. Pipeline Parallel

### Megatron-LM 如何实现

`megatron/core/pipeline_parallel/schedules.py` 根据 PP/virtual PP 配置选择无 pipeline、
非交错 1F1B 或交错 schedule。训练 iteration 被拆成多个 micro-batch，并经历：

1. warmup：先填充 pipeline。
2. steady state：交替执行 forward/backward，即 1F1B。
3. cooldown：排空尚未完成的 backward。

stage 间通过 P2P 发送 activation 和 activation gradient。bubble 取决于 stage 数、
micro-batch 数和各 stage 计算是否均衡；virtual pipeline 可缩短 bubble，但会增加调度和
通信复杂度。最后一个 stage 计算 loss，embedding 等共享参数还可能需要额外同步。

### MiniTrainBench 当前如何实现

当前 `Trainer` 在每个 rank 上执行完整模型，gradient accumulation 只控制 micro-batch
梯度同步，不切 stage，也没有 activation P2P 或 1F1B schedule。

### 差距和边界

Gradient accumulation 不等于 Pipeline Parallel。MiniTrainBench 没有 layer partition、
schedule、P2P buffer、stage-local loss 或 bubble 统计。本轮通过 Megatron PP=2/4 的外部
实验观察 pipeline 行为，不自行实现不完整 PP runtime。

## 4. Distributed Optimizer

### Megatron-LM 如何实现

Megatron distributed optimizer 在 data-parallel ranks 间切分 optimizer state，并使用
contiguous parameter/gradient buffers。gradient 进入 main gradient buffer 后通过
reduce-scatter 分片；optimizer 只更新本 rank 的 shard，随后按需 all-gather 参数。
`overlap-grad-reduce` 和 `overlap-param-gather` 尝试把通信隐藏在 backward/forward 中。

它与 ZeRO-1/2/3 的关系不能仅按名字判断：应分别说明 parameter、gradient、optimizer
state 哪些是 replicated、哪些是 sharded，以及参数 gather 在何时发生。FSDP 则以模块
包装和参数 handle 为边界，执行参数 all-gather 与梯度 reduce-scatter。

### MiniTrainBench 当前如何实现

DDP 使用普通 AdamW 完整副本；FSDP 由 PyTorch 负责参数、梯度和 optimizer state 的
sharded state；DeepSpeed ZeRO-2/3 通过独立 adapter 横向 benchmark。现有报告比较吞吐、
step time 和峰值显存，但 23.2M 小模型容易被 engine 固定开销主导。

### 差距和边界

MiniTrainBench 没有 contiguous grad buffer、bucket overlap 控制、参数预取或自定义
distributed optimizer。显存压力矩阵用于寻找“DDP 更快但放不下、FSDP/ZeRO 更慢但可训练”
的规模边界，避免从小模型直接外推大模型结论。

## 5. Distributed Checkpoint

### Megatron-LM 如何实现

`megatron/core/dist_checkpointing/` 让模型暴露 sharded state dict，并在保存时记录 shard
placement。加载时可根据目标 TP/PP 拓扑重新映射部分 shard，这比“同 world size 原样恢复”
更接近生产训练中的并行度转换。完整训练恢复还涉及 optimizer、scheduler、iteration、
RNG 和数据迭代状态。

### MiniTrainBench 当前如何实现

Checkpoint v3 使用 PyTorch DCP 保存 DDP/FSDP model/optimizer/scheduler，保存 TrainState 和
每 rank CPU/CUDA RNG，并采用临时目录、barrier、`READY`、`latest` 和 retention。
`checkpoint verify` 能比较 model、optimizer、scheduler、TrainState 与 RNG digest，证明
同配置恢复的一致性。

### 差距和边界

当前明确要求 strategy、precision、world size、rank mapping 和模型配置一致，不支持
TP/PP shard 或跨 world size reshard。MiniTrainBench 的优势是发布生命周期和精确恢复
可验证；Megatron 的优势是面向模型并行拓扑的 sharded state 与重映射能力。

## 外部实验要回答的问题

`scripts/run_megatron_tp_pp_matrix.sh` 固定上游 ref，并对同一 8 卡节点运行 DP baseline、
TP=2/4、TP=2+PP=2 和 PP=4。实验重点不是证明 Megatron 快，而是回答：

- TP 增大后，单 rank 显存是否下降，层内 collective 是否抬高 step time。
- PP 增大后，micro-batch 数是否足以摊薄 bubble。
- TP/PP 组合时 DP degree 和 global batch 是否仍保持公平。
- 日志中哪些指标可以直接观测，哪些必须依赖 trace/TensorBoard，不能凭均值推断。

源码不复制进仓库，结果必须记录 Megatron commit、容器、完整命令和失败原因。

## 本轮 8 卡兼容性结果

固定 `core_v0.18.2` commit 后，`TP/PP/DP=1/1/8`、`2/1/4`、`4/1/2`、`2/2/2`、
`1/4/2` 五组 topology 均完成 forward/backward/optimizer smoke。这证明外部 runner 的
parallel group、TP、PP 和 distributed optimizer 参数组合可执行，但不等于完成正式性能
benchmark。

本轮 NGC 大镜像未能在合理时间内拉取，且 8 张 GPU 上存在其他计算进程，因此使用锁定的
官方 PyTorch fallback 做兼容性验证。fallback 缺少 Transformer Engine/APEX fused kernels，
显式关闭 RoPE、persistent LayerNorm、weight-gradient 和 masked-softmax fusion；torch
LayerNorm 不支持 sequence parallel，因此 TP 配置的 SP 状态记录为 false。报告将
`performance_valid` 标为 false，并隐藏吞吐和显存值。

PP=2 和 PP=4 分别记录 0.2 与约 0.429 的 fill-drain 理论 bubble proxy；没有 pipeline trace，
所以不能声称观察到同等比例的 idle。正式结论仍要求 NGC、独占 GPU、repeat=3 和完整
20-step 测量协议，当前结果只回答“能否按这些拓扑运行”。

## 上游阅读入口

- [Megatron Core QuickStart](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/QuickStart.md)
- [parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/parallel_state.py)
- [pipeline schedules](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/pipeline_parallel/schedules.py)
- [distributed optimizer](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/docs/user-guide/features/dist_optimizer.md)
- [distributed checkpointing](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.18.2/megatron/core/dist_checkpointing)
