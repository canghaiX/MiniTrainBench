# 8 卡 A100 Profiler Case Study

本文对应 `results/profile_8gpu/`。原始 Chrome trace 默认不提交，只提交每 rank summary、
跨 rank 统计和可复现命令。

## 实验口径

- 单节点 8x A100-SXM4-40GB。
- DDP/FSDP 使用相同 23.2M GPT-like 模型、BF16、sequence length 256。
- `grad_accum_steps=4`；DDP `auto` 解析为末个 micro-batch 同步，FSDP `auto` 解析为
  每个 micro-batch 同步。
- Profiler 只用于定位，不将 profile step time 与普通 benchmark 吞吐直接比较。

## 分析方法

1. 先看 step breakdown，判断 data、forward/backward、optimizer 哪段主导。
2. 再看每 rank collective event，区分 DDP all-reduce 与 FSDP all-gather/reduce-scatter。
3. 用 rank min/p50/max 和 `max/p50` 判断是否存在明显 straggler。
4. `key_averages()` 不能证明 overlap；只有 Chrome trace 中计算 kernel 与 NCCL kernel
   在不同 stream 上真实重叠，才能写成“观察到 overlap”。

## 实测结论

本次实测使用 PyTorch 2.10.0+cu130、8x A100-SXM4-40GB。DDP 的平均 step time 为
99.11 ms，FSDP 为 199.49 ms；两者的 data time 均低于 0.23 ms，瓶颈不在 synthetic
data 生成。DDP 的 optimizer step 为 2.84 ms，FSDP 为 1.13 ms，主要差异落在
forward/backward 及其伴随的 collective。

| Strategy | Step (ms) | Forward/backward (ms) | Collective/step (ms) | Peak memory/rank (MB) | Rank max/p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDP | 99.11 | 95.87 | 7.33 | 569.65 | 1.0018 |
| FSDP | 199.49 | 198.03 | 136.91 | 131.83-134.30 | 1.0007 |

DDP 的摘要记录到每 3 个 profile step 共 12 次 `nccl:all_reduce`；FSDP 记录到
156 次 all-gather 和 84 次 reduce-scatter。collective/step 是各 rank 的 NCCL event
duration 累加后再除以 3，不等同于不可重叠的通信 wall time。两个策略的 rank
`max/p50` 都接近 1.0，本轮没有观察到 step-time straggler。

原始 trace 在本机生成但默认不提交；当前自动摘要仍将 overlap 标记为“未确定”，因为
`key_averages()` 无法保留跨 CUDA stream 的时序关系。这里不根据 op duration 与 step
duration 的差值推断 overlap。

ZeRO-3 本轮不采集独立 trace。`results/zero_repeat3/zero3_8gpu.json` 的独立实验为
46.8K tokens/sec、87.56 ms、约 2.16 GB；该实验没有使用本页 `grad_accum_steps=4`
的 Profiler 口径，只能说明既有 benchmark 结果，不能与上表直接做性能归因，也不能
据此声称参数 gather 在时间线上与计算发生了重叠。

MoE equal/uneven all-to-all 使用 `results/moe_comm/` 的 microbenchmark 解释链路与负载
不均，不与 dense DDP/FSDP profiler 合并成同一占比。

## 结论边界

23.2M 模型的 kernel 很短，Python、launcher 和 collective 固定延迟容易放大，通信占比
也更容易抖动。显存压力矩阵中的 medium/large 档位更适合判断 FSDP/ZeRO 的真实价值；
该 Profiler case study 主要证明性能定位方法和 collective 语义。
