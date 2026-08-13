# MiniTrainBench Profiler 摘要

- Strategy：fsdp
- GPU 数：8
- 精度：bf16
- Trace 目录：`/workspace/results/profile_8gpu/fsdp_8gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.22 | 0.02 | 0.21 | 0.25 |
| forward_backward_ms | 198.03 | 3.66 | 193.33 | 202.26 |
| optimizer_step_ms | 1.13 | 0.15 | 1.02 | 1.35 |
| step_time_ms | 199.49 | 3.50 | 195.04 | 203.58 |
| tokens_per_sec | 41078.15 | 721.73 | 40239.20 | 42001.08 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 199.49 |
| Step p50 (ms) | 200.02 |
| Step max (ms) | 200.17 |
| Straggler ratio (max/p50) | 1.001 |
| 每 rank collective total (ms, mean) | 410.72 |
| 每 rank collective/step (ms, mean) | 136.91 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 594.49 |
| CUDA | record_param_comms | 240 | 307.91 |
| CUDA | FullyShardedDataParallel.forward | 84 | 240.94 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 166.82 |
| CUDA | nccl:_all_gather_base | 156 | 166.82 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 147.39 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 141.09 |
| CUDA | nccl:_reduce_scatter_base | 84 | 141.09 |
| CPU | ProfilerStep* | 3 | 355.10 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 51.74 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 43.40 |
| CPU | cudaLaunchKernel | 5115 | 43.28 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 41.55 |
| CPU | FullyShardedDataParallel.forward | 84 | 33.10 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 25.15 |
| CPU | cudaStreamSynchronize | 12 | 17.02 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 166.82 |
| Collective | nccl:_all_gather_base | 156 | 166.82 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 141.09 |
| Collective | nccl:_reduce_scatter_base | 84 | 141.09 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.38 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 593.99 |
| CUDA | FullyShardedDataParallel.forward | 84 | 271.09 |
| CUDA | record_param_comms | 240 | 168.10 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 93.43 |
| CUDA | nccl:_all_gather_base | 156 | 93.43 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 83.88 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 74.67 |
| CUDA | nccl:_reduce_scatter_base | 84 | 74.67 |
| CPU | ProfilerStep* | 3 | 357.85 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 54.28 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 45.05 |
| CPU | cudaLaunchKernel | 5115 | 43.63 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 43.11 |
| CPU | FullyShardedDataParallel.forward | 84 | 30.26 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 27.80 |
| CPU | record_param_comms | 240 | 14.41 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 93.43 |
| Collective | nccl:_all_gather_base | 156 | 93.43 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 74.67 |
| Collective | nccl:_reduce_scatter_base | 84 | 74.67 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.29 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 2

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 596.03 |
| CUDA | record_param_comms | 240 | 464.60 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 249.66 |
| CUDA | nccl:_all_gather_base | 156 | 249.66 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 220.35 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 214.94 |
| CUDA | nccl:_reduce_scatter_base | 84 | 214.94 |
| CUDA | FullyShardedDataParallel.forward | 84 | 209.69 |
| CPU | ProfilerStep* | 3 | 348.56 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 50.82 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 44.35 |
| CPU | cudaLaunchKernel | 5115 | 41.66 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 41.17 |
| CPU | cudaStreamSynchronize | 12 | 34.66 |
| CPU | FullyShardedDataParallel.forward | 84 | 27.84 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 25.97 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 249.66 |
| Collective | nccl:_all_gather_base | 156 | 249.66 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 214.94 |
| Collective | nccl:_reduce_scatter_base | 84 | 214.94 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.48 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 3

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 597.31 |
| CUDA | record_param_comms | 240 | 548.81 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 287.99 |
| CUDA | nccl:_all_gather_base | 156 | 287.99 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 263.42 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 260.82 |
| CUDA | nccl:_reduce_scatter_base | 84 | 260.82 |
| CUDA | FullyShardedDataParallel.forward | 84 | 199.52 |
| CPU | ProfilerStep* | 3 | 337.67 |
| CPU | cudaStreamSynchronize | 12 | 54.05 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 46.57 |
| CPU | cudaLaunchKernel | 5115 | 42.71 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 41.91 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 38.98 |
| CPU | FullyShardedDataParallel.forward | 84 | 27.50 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 24.39 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 287.99 |
| Collective | nccl:_all_gather_base | 156 | 287.99 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 260.82 |
| Collective | nccl:_reduce_scatter_base | 84 | 260.82 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.29 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 4

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 597.53 |
| CUDA | record_param_comms | 240 | 560.00 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 297.85 |
| CUDA | nccl:_all_gather_base | 156 | 297.85 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 264.27 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 262.15 |
| CUDA | nccl:_reduce_scatter_base | 84 | 262.15 |
| CUDA | FullyShardedDataParallel.forward | 84 | 197.34 |
| CPU | ProfilerStep* | 3 | 337.47 |
| CPU | cudaStreamSynchronize | 12 | 55.54 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 46.77 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 41.99 |
| CPU | cudaLaunchKernel | 5115 | 41.33 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 39.41 |
| CPU | FullyShardedDataParallel.forward | 84 | 27.23 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 23.80 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 297.85 |
| Collective | nccl:_all_gather_base | 156 | 297.85 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 262.15 |
| Collective | nccl:_reduce_scatter_base | 84 | 262.15 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.30 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 5

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 596.81 |
| CUDA | record_param_comms | 240 | 518.38 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 275.14 |
| CUDA | nccl:_all_gather_base | 156 | 275.14 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 247.24 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 243.24 |
| CUDA | nccl:_reduce_scatter_base | 84 | 243.24 |
| CUDA | FullyShardedDataParallel.forward | 84 | 201.05 |
| CPU | ProfilerStep* | 3 | 344.54 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 48.11 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 44.35 |
| CPU | cudaLaunchKernel | 5115 | 41.90 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 40.91 |
| CPU | cudaStreamSynchronize | 12 | 40.56 |
| CPU | FullyShardedDataParallel.forward | 84 | 29.03 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 26.52 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 275.14 |
| Collective | nccl:_all_gather_base | 156 | 275.14 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 243.24 |
| Collective | nccl:_reduce_scatter_base | 84 | 243.24 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.31 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 6

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 594.98 |
| CUDA | record_param_comms | 240 | 333.70 |
| CUDA | FullyShardedDataParallel.forward | 84 | 228.59 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 189.27 |
| CUDA | nccl:_all_gather_base | 156 | 189.27 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 151.74 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 144.43 |
| CUDA | nccl:_reduce_scatter_base | 84 | 144.43 |
| CPU | ProfilerStep* | 3 | 360.93 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 51.54 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 44.46 |
| CPU | cudaLaunchKernel | 5115 | 43.59 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 42.36 |
| CPU | FullyShardedDataParallel.forward | 84 | 28.14 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 24.78 |
| CPU | cudaStreamSynchronize | 12 | 15.40 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 189.27 |
| Collective | nccl:_all_gather_base | 156 | 189.27 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 144.43 |
| Collective | nccl:_reduce_scatter_base | 84 | 144.43 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.34 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

### Rank 7

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 594.33 |
| CUDA | record_param_comms | 240 | 384.30 |
| CUDA | FullyShardedDataParallel.forward | 84 | 219.85 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 217.19 |
| CUDA | nccl:_all_gather_base | 156 | 217.19 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 173.02 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 167.10 |
| CUDA | nccl:_reduce_scatter_base | 84 | 167.10 |
| CPU | ProfilerStep* | 3 | 352.37 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 51.71 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 48.51 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 43.17 |
| CPU | cudaLaunchKernel | 5115 | 42.92 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 29.46 |
| CPU | FullyShardedDataParallel.forward | 84 | 27.98 |
| CPU | cudaStreamSynchronize | 12 | 20.88 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 217.19 |
| Collective | nccl:_all_gather_base | 156 | 217.19 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 167.10 |
| Collective | nccl:_reduce_scatter_base | 84 | 167.10 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.32 |
| Collective | nccl:_all_gather_base | 156 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 84 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
