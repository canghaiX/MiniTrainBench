# MoE 与 Expert Parallel 训练笔记

本文档说明 MiniTrainBench 为什么补 `all_to_all` benchmark，以及它和 MoE 训练
Runtime 的关系。当前项目不实现完整 MoE layer，但提供 top-1 routing、capacity、
overflow、dispatch/combine 和负载不均衡的 toy demo。

## MoE 前向路径

MoE layer 通常由 router、若干 expert 和 token combine 组成。router 根据 token 的
hidden state 计算每个 expert 的打分，然后选择 top-k expert。被选中的 token 会被
dispatch 到对应 expert，expert 完成 MLP 计算后，再按 router weight combine 回原来的
token 顺序。

在 expert parallel 中，不同 rank 持有不同 expert。一个 rank 上的 token 可能需要发送到
其他 rank 的 expert，其他 rank 也可能把 token 发回来。因此 MoE 的核心通信不是 DDP 的
gradient all-reduce，而是 forward 和 backward 中围绕 token dispatch/combine 的
`all_to_all`。

## Router、Top-k 与 Load Balancing

- router 输出每个 token 到每个 expert 的 logits。
- top-k routing 只选择得分最高的 k 个 expert，常见配置是 top-1 或 top-2。
- capacity factor 限制每个 expert 最多接收多少 token，避免某个 expert 被打爆。
- load balancing loss 鼓励 token 分布更均匀，降低 expert 间负载不均。
- overflow token 需要明确策略：丢弃、走残差路径、或 fallback 到其他 expert。

Router 质量会直接影响系统性能。即使总 token 数固定，如果某些 expert 收到过多 token，
对应 rank 就会成为 straggler；同步训练的 step time 由最慢 rank 决定。

## All-to-all 为什么关键

MoE token dispatch 的通信形态天然是 all-to-all：每个 rank 都可能向每个其他 rank 发送
不同数量的 token。等长 split 可以模拟理想负载均衡；非等长 split 更接近真实 router
产生的 token 分布。

MiniTrainBench 的 `minitrainbench comm --operations all_to_all` 同时支持：

- `equal`：每个 peer 收发相同元素数，用于看链路和 collective 的理想带宽。
- `uneven`：每个 rank 到不同 peer 的 split 不同，用于模拟 MoE token dispatch 的
  shape 压力。
- `both`：默认同时跑两种模式，便于比较 latency 和 bandwidth。

`minitrainbench moe route` 进一步把通信 benchmark 前的 routing 逻辑串起来：

- 根据 router logits 做 top-1 expert 选择。
- 按 capacity factor 截断每个 expert 的 token。
- 统计 expert load、overflow、load-balance loss 和 owner rank。
- 使用 `all_to_all_single` 做 toy token dispatch/combine；当前不实现 expert grouped
  GEMM、完整 residual combine 或训练 loss 回传。

## 训练框架实现要点

- 先根据 top-k 结果统计每个 destination expert/rank 的 token 数。
- 用 prefix sum 把 token pack 成按目标 rank 分段的连续 buffer。
- 调用 `all_to_all_single` 完成跨 rank token dispatch。
- 每个 rank 对本地 expert batch 做 grouped GEMM 或按 expert 分组计算。
- 再通过一次 all-to-all 把 expert output combine 回原 token owner。
- backward 需要沿相反路径传播 expert output、router weight 和 token hidden state 的梯度。

真实系统还需要处理 padding、capacity overflow、expert placement、grouped GEMM、
router auxiliary loss、token permutation kernel 和跨节点拓扑。但这些都建立在
`all_to_all` 通信语义之上。

## 面试表达

可以把本项目这样讲：

> 我没有把完整 MoE layer 塞进 benchmark，但补了 MoE expert parallel 最核心的
> all-to-all microbenchmark，并区分 equal/uneven split。这样可以从通信层解释 token
> dispatch/combine 的瓶颈，也能和 DDP/FSDP/ZeRO 的 all-reduce、all-gather、
> reduce-scatter 形成对照。
