# Megatron-style 并行训练笔记

本文档记录 Tensor Parallel、Pipeline Parallel 和 Sequence Parallel 的基本语义。项目当前
只实现 toy tensor parallel correctness check，不实现完整 TP/PP/SP 训练 Runtime。

## Tensor Parallel

Tensor Parallel 将单层内部的矩阵乘法切到多个 rank 上。Megatron-style TP 里最常见的是
`ColumnParallelLinear` 和 `RowParallelLinear`。

`ColumnParallelLinear` 按输出维切分权重：

```text
Y = X @ W
W = [W0, W1, ..., Wn]
Yi = X @ Wi
Y = concat(Y0, Y1, ..., Yn)
```

它适合切 MLP 的 up/gate projection 或 attention 的 QKV/head 维度。forward 只产生本地
输出 shard；如果下一层能消费 shard，就可以延迟 all-gather。

`RowParallelLinear` 按输入维切分权重：

```text
X = [X0, X1, ..., Xn]
W = [W0, W1, ..., Wn]
Y = sum_i Xi @ Wi
```

每个 rank 计算局部 partial output，然后 all-reduce 得到完整输出。MLP 中常见模式是
先 column split 扩展 hidden，再 row split 投回 hidden size；这样中间 activation 可以
在 TP rank 间保持切分，减少不必要的 gather。

MiniTrainBench 的 `minitrainbench tp check` 会把 toy `ColumnParallelLinear` 和
`RowParallelLinear` 与单卡 reference 对齐，比较 forward、input grad、weight grad 和
bias grad 的最大误差。

## Pipeline Parallel

Pipeline Parallel 按 layer/stage 切分模型。每个 stage 只持有一段层，micro-batch 在
stage 间流动。它解决的是单卡放不下全部层的问题，但会引入 pipeline bubble。

Bubble 来自流水线填充和排空：第一个 stage 开始计算时，后面的 stage 还没有输入；最后
一个 micro-batch 离开前，前面的 stage 已经空闲。micro-batch 数越少，bubble 占比越高。
1F1B schedule 通过一个 forward 后接一个 backward 的方式降低 activation 驻留时间和空泡。

本项目暂不实现 PP schedule，因为正确实现还需要跨 stage send/recv、activation 保存、
loss stage、反向依赖和 checkpoint/recompute 策略。

## Sequence Parallel

Sequence Parallel 通常和 Tensor Parallel 配套。TP 已经把部分线性层按 hidden 维切分，
但 LayerNorm、Dropout 和 residual 等操作可能仍保留完整 sequence activation。SP 将部分
activation 沿 sequence 维分片，降低每个 TP rank 的激活显存。

SP 的代价是额外的 all-gather/reduce-scatter，以及对随机性、LayerNorm 统计和 dropout
mask 的更严格管理。它适合长序列和大模型训练，但不适合作为本项目当前的最小实现。

## 与本项目现有能力的关系

- DDP 主要依赖 gradient all-reduce。
- FSDP 主要依赖参数 all-gather 和梯度 reduce-scatter。
- ZeRO-2/3 分别分片 optimizer/gradient 和 parameter state。
- MoE expert parallel 依赖 all-to-all token dispatch/combine。
- Megatron TP 使用 intra-layer all-reduce/all-gather/reduce-scatter 来组合 shard。

这些并行维度可以组合成 3D/4D parallelism。MiniTrainBench 的定位不是复刻完整
Megatron，而是用小而可验证的 demo 说明每类并行的通信语义和工程边界。
