# MiniTrainBench Profiler 摘要

- Strategy：fsdp
- GPU 数：2
- 精度：bf16
- Trace 目录：`/workspace/results/profile/fsdp_2gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.21 | 0.02 | 0.19 | 0.23 |
| forward_backward_ms | 178.30 | 3.78 | 174.03 | 183.23 |
| optimizer_step_ms | 5.48 | 0.16 | 5.26 | 5.66 |
| step_time_ms | 184.08 | 3.86 | 179.83 | 189.17 |
| tokens_per_sec | 11130.62 | 231.78 | 10826.38 | 11388.40 |
| grad_norm | 134.31 | 27.23 | 95.82 | 154.41 |
| learning_rate | 0.00 | 0.00 | 0.00 | 0.00 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 184.08 |
| Step p50 (ms) | 184.15 |
| Step max (ms) | 184.22 |
| Straggler ratio (max/p50) | 1.000 |
| 每 rank collective total (ms, mean) | 80.43 |
| 每 rank collective/step (ms, mean) | 26.81 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 547.60 |
| CUDA | FullyShardedDataParallel.forward | 84 | 258.73 |
| CUDA | record_param_comms | 246 | 109.30 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 62.00 |
| CUDA | nccl:_all_gather_base | 156 | 62.00 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 52.65 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 42.17 |
| CUDA | nccl:_reduce_scatter_base | 84 | 42.17 |
| CPU | ProfilerStep* | 3 | 330.52 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 46.14 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 42.85 |
| CPU | cudaLaunchKernel | 5322 | 42.77 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 37.16 |
| CPU | FullyShardedDataParallel.forward | 84 | 26.36 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 26.07 |
| CPU | record_param_comms | 246 | 13.87 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 62.00 |
| Collective | nccl:_all_gather_base | 156 | 62.00 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 42.17 |
| Collective | nccl:_reduce_scatter_base | 84 | 42.17 |
| Collective | nccl:all_reduce | 6 | 5.12 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.62 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.51 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.28 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 546.55 |
| CUDA | FullyShardedDataParallel.forward | 84 | 258.89 |
| CUDA | record_param_comms | 246 | 51.57 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 33.00 |
| CUDA | nccl:_all_gather_base | 156 | 33.00 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 28.31 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 18.50 |
| CUDA | nccl:_reduce_scatter_base | 84 | 18.50 |
| CPU | ProfilerStep* | 3 | 334.20 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 45.62 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 45.05 |
| CPU | cudaLaunchKernel | 5454 | 44.01 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 38.40 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 28.43 |
| CPU | FullyShardedDataParallel.forward | 84 | 25.97 |
| CPU | record_param_comms | 246 | 13.25 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 33.00 |
| Collective | nccl:_all_gather_base | 156 | 33.00 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 18.50 |
| Collective | nccl:_reduce_scatter_base | 84 | 18.50 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.31 |
| Collective | nccl:all_reduce | 6 | 0.07 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.05 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.03 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
