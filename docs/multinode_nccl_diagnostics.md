# 多机与 NCCL 诊断笔记

本文记录 MiniTrainBench 在多机训练里最需要先排查的几类问题。当前仓库不提交真实多机实测结果，但提供 `doctor`、torchrun 多机模板和诊断清单，方便面试时解释你怎么定位分布式问题。

## 先看什么

1. GPU 是否可见，`CUDA_VISIBLE_DEVICES` 是否和预期一致。
2. `MASTER_ADDR` / `MASTER_PORT` 是否对所有节点一致且可达。
3. `WORLD_SIZE` 是否等于 `nnodes * nproc_per_node`。
4. `LOCAL_RANK` 是否没有越界。
5. `NCCL_SOCKET_IFNAME` 是否选中了正确网卡。
6. `NCCL_IB_DISABLE`、`NCCL_ASYNC_ERROR_HANDLING`、`TORCH_NCCL_ASYNC_ERROR_HANDLING` 是否符合当前网络环境。

## 常见 hang 原因

- 节点间端口不通，或者 `MASTER_PORT` 被别的进程占用。
- `WORLD_SIZE` / `RANK` 配错，导致某些 rank 在等永远不会到的伙伴。
- 容器网络没打通，或者 `NCCL_SOCKET_IFNAME` 选错网卡。
- IB/GDR 环境不稳定，NCCL 退回到较慢路径或直接超时。
- 不同节点的 PyTorch / CUDA / NCCL 版本不一致。

## 排障顺序

1. 先跑 `minitrainbench doctor`，确认环境、网卡和 rendezvous 端口。
2. 再跑最小 all-reduce / all-gather / all-to-all benchmark，验证 collective 是否能正常完成。
3. 最后再跑训练 Runtime，看是计算、通信还是 checkpoint/resume 语义出问题。

## 讲给面试官听时怎么说

> 多机分布式问题通常先不是模型，而是环境、rendezvous、网卡和 NCCL 配置。
> 我在 MiniTrainBench 里补了 doctor 和多机 torchrun 模板，先把 GPU、端口、网卡和 NCCL 环境检查出来，再去看 collective 和训练 step time。这样定位顺序和真实训练平台更接近。
