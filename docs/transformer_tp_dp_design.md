# Transformer TP/DP 设计说明

## 进程组和参数布局

`tp transformer-train` 沿用现有的 rank layout：

```text
tp_rank = global_rank % tp_size
dp_rank = global_rank // tp_size
```

同一个 `dp_rank` 的进程组成 TP group，同一个 `tp_rank` 的进程组成 DP
group。Column Parallel Q/K/V 和 MLP up projection 沿输出维度切分；attention
output projection 和 MLP down projection 沿输入维度切分，并在 TP group 内对
局部输出做 all-reduce。

LayerNorm、attention output bias 和 MLP output bias 是 TP-replicated 参数。它们
的梯度先在 TP group 内 all-reduce，再对所有 sharded/replicated 参数统一做 DP
gradient averaging。这样每个 DP replica 的 optimizer step 使用相同的全局梯度，
同时不会把 TP-sharded 参数错误地复制到其他 TP rank。

## Correctness probe

训练前用同一组完整 FP32 reference 权重执行单卡 pre-norm Transformer Block：

1. LayerNorm → column-parallel QKV → TP-local causal MHA；
2. row-parallel attention output → residual；
3. LayerNorm → column/row-parallel GELU MLP → residual。

probe 比较 forward max error、MSE loss error，以及 Q/K/V、attention output、MLP
和两组 LayerNorm 的局部参数梯度。FP32 使用 absolute tolerance；BF16 使用
relative L2 tolerance。输入由 `dp_rank` 和 step 派生的 deterministic seed 生成，
因此 DP rank 使用不同 batch，同时 TP rank 在同一个 DP replica 内看到相同 batch。

## Benchmark 口径和边界

输出包含 TP/DP degree、step P50/P95、tokens/sec、TP/DP communication bytes
estimate 和 correctness 结果。通信量是 activation all-reduce 与 gradient
all-reduce 的估算，不代表 NCCL 实际链路计数。

本入口当前只覆盖 MHA、单个 Transformer Block 和 regression target；不包含
vocab-parallel cross entropy、checkpoint/resharding、optimizer state sharding、
pipeline parallel 或多机 TP。
