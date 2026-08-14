# MiniTrainBench Profiler 汇总

## ddp_8gpu

# MiniTrainBench Profiler 摘要

- Strategy：ddp
- GPU 数：8
- 精度：bf16
- Trace 目录：`/workspace/results/profile_8gpu/ddp_8gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.21 | 0.01 | 0.20 | 0.23 |
| forward_backward_ms | 96.89 | 1.10 | 95.42 | 98.07 |
| optimizer_step_ms | 5.48 | 0.13 | 5.38 | 5.67 |
| step_time_ms | 102.79 | 1.20 | 101.24 | 104.15 |
| tokens_per_sec | 79704.47 | 929.40 | 78655.56 | 80914.87 |
| grad_norm | 107.46 | 29.80 | 79.05 | 148.62 |
| learning_rate | 0.00 | 0.00 | 0.00 | 0.00 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 102.75 |
| Step p50 (ms) | 102.78 |
| Step max (ms) | 102.88 |
| Straggler ratio (max/p50) | 1.001 |
| 每 rank collective total (ms, mean) | 40.34 |
| 每 rank collective/step (ms, mean) | 13.45 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 299.68 |
| CUDA | DistributedDataParallel.forward | 12 | 88.15 |
| CUDA | record_param_comms | 21 | 33.16 |
| CUDA | nccl:all_reduce | 15 | 33.11 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 30.85 |
| CUDA | aten::copy_ | 2262 | 12.05 |
| CUDA | aten::mm | 612 | 7.83 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.16 |
| CPU | ProfilerStep* | 3 | 195.43 |
| CPU | cudaLaunchKernel | 5997 | 43.90 |
| CPU | DistributedDataParallel.forward | 12 | 21.34 |
| CPU | aten::copy_ | 2262 | 13.40 |
| CPU | aten::empty_strided | 2259 | 12.30 |
| CPU | aten::mm | 612 | 11.69 |
| CPU | aten::empty | 1839 | 9.23 |
| CPU | aten::addmm | 288 | 8.53 |
| Collective | nccl:all_reduce | 15 | 33.11 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 30.85 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.26 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.05 |
| Collective | nccl:broadcast | 3 | 0.05 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 299.84 |
| CUDA | DistributedDataParallel.forward | 12 | 87.64 |
| CUDA | record_param_comms | 21 | 20.63 |
| CUDA | nccl:all_reduce | 15 | 20.34 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 18.62 |
| CUDA | aten::copy_ | 2262 | 12.02 |
| CUDA | aten::mm | 612 | 7.68 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.08 |
| CPU | ProfilerStep* | 3 | 199.60 |
| CPU | cudaLaunchKernel | 5997 | 44.41 |
| CPU | DistributedDataParallel.forward | 12 | 20.83 |
| CPU | aten::copy_ | 2262 | 13.38 |
| CPU | aten::empty_strided | 2259 | 12.32 |
| CPU | aten::mm | 612 | 11.90 |
| CPU | aten::empty | 1839 | 9.29 |
| CPU | aten::addmm | 288 | 8.63 |
| Collective | nccl:all_reduce | 15 | 20.34 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 18.62 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 1.71 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.29 |
| Collective | nccl:broadcast | 3 | 0.29 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 2

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 299.97 |
| CUDA | DistributedDataParallel.forward | 12 | 87.58 |
| CUDA | record_param_comms | 21 | 36.12 |
| CUDA | nccl:all_reduce | 15 | 35.59 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 32.97 |
| CUDA | aten::copy_ | 2262 | 11.89 |
| CUDA | aten::mm | 612 | 7.71 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.06 |
| CPU | ProfilerStep* | 3 | 195.97 |
| CPU | cudaLaunchKernel | 5997 | 44.45 |
| CPU | DistributedDataParallel.forward | 12 | 20.72 |
| CPU | aten::copy_ | 2262 | 13.24 |
| CPU | aten::empty_strided | 2259 | 12.38 |
| CPU | aten::mm | 612 | 11.73 |
| CPU | aten::empty | 1839 | 9.35 |
| CPU | aten::addmm | 288 | 8.55 |
| Collective | nccl:all_reduce | 15 | 35.59 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 32.97 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.62 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.53 |
| Collective | nccl:broadcast | 3 | 0.53 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 3

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 300.04 |
| CUDA | DistributedDataParallel.forward | 12 | 86.28 |
| CUDA | record_param_comms | 21 | 47.61 |
| CUDA | nccl:all_reduce | 15 | 46.99 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 44.16 |
| CUDA | aten::copy_ | 2262 | 11.93 |
| CUDA | aten::mm | 612 | 7.85 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.01 |
| CPU | ProfilerStep* | 3 | 194.08 |
| CPU | cudaLaunchKernel | 5997 | 43.40 |
| CPU | DistributedDataParallel.forward | 12 | 20.61 |
| CPU | aten::copy_ | 2262 | 13.25 |
| CPU | aten::empty_strided | 2259 | 12.24 |
| CPU | aten::mm | 612 | 11.68 |
| CPU | cudaStreamSynchronize | 18 | 11.23 |
| CPU | aten::empty | 1839 | 9.24 |
| Collective | nccl:all_reduce | 15 | 46.99 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 44.16 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.83 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.61 |
| Collective | nccl:broadcast | 3 | 0.61 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 4

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 300.12 |
| CUDA | DistributedDataParallel.forward | 12 | 87.37 |
| CUDA | record_param_comms | 21 | 34.26 |
| CUDA | nccl:all_reduce | 15 | 33.61 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 31.66 |
| CUDA | aten::copy_ | 2262 | 11.87 |
| CUDA | aten::mm | 612 | 7.70 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.01 |
| CPU | ProfilerStep* | 3 | 192.63 |
| CPU | cudaLaunchKernel | 5997 | 42.21 |
| CPU | DistributedDataParallel.forward | 12 | 21.03 |
| CPU | aten::copy_ | 2262 | 13.38 |
| CPU | aten::empty_strided | 2259 | 12.25 |
| CPU | aten::mm | 612 | 11.83 |
| CPU | aten::empty | 1839 | 9.35 |
| CPU | aten::addmm | 288 | 8.53 |
| Collective | nccl:all_reduce | 15 | 33.61 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 31.66 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 1.96 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.65 |
| Collective | nccl:broadcast | 3 | 0.65 |
| Collective | c10d::broadcast_ | 3 | 0.07 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 5

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 300.04 |
| CUDA | DistributedDataParallel.forward | 12 | 86.26 |
| CUDA | record_param_comms | 21 | 49.43 |
| CUDA | nccl:all_reduce | 15 | 48.80 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 46.04 |
| CUDA | aten::copy_ | 2262 | 11.83 |
| CUDA | aten::mm | 612 | 7.60 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.02 |
| CPU | ProfilerStep* | 3 | 193.60 |
| CPU | cudaLaunchKernel | 5997 | 43.73 |
| CPU | DistributedDataParallel.forward | 12 | 20.53 |
| CPU | aten::copy_ | 2262 | 13.26 |
| CPU | aten::empty_strided | 2259 | 12.13 |
| CPU | cudaStreamSynchronize | 18 | 11.95 |
| CPU | aten::mm | 612 | 11.67 |
| CPU | aten::empty | 1839 | 9.25 |
| Collective | nccl:all_reduce | 15 | 48.80 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 46.04 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.76 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.64 |
| Collective | nccl:broadcast | 3 | 0.64 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 6

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 300.38 |
| CUDA | DistributedDataParallel.forward | 12 | 85.27 |
| CUDA | record_param_comms | 21 | 60.66 |
| CUDA | nccl:all_reduce | 15 | 59.61 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 56.51 |
| CUDA | aten::copy_ | 2262 | 11.26 |
| CUDA | aten::mm | 612 | 7.31 |
| CUDA | Optimizer.step#AdamW.step | 3 | 5.96 |
| CPU | ProfilerStep* | 3 | 190.75 |
| CPU | cudaLaunchKernel | 5997 | 43.89 |
| CPU | DistributedDataParallel.forward | 12 | 21.03 |
| CPU | cudaStreamSynchronize | 18 | 15.66 |
| CPU | aten::copy_ | 2262 | 13.50 |
| CPU | aten::empty_strided | 2259 | 12.04 |
| CPU | aten::mm | 612 | 11.67 |
| CPU | aten::empty | 1839 | 9.04 |
| Collective | nccl:all_reduce | 15 | 59.61 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 56.51 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 3.10 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 1.05 |
| Collective | nccl:broadcast | 3 | 1.05 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 7

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 300.06 |
| CUDA | DistributedDataParallel.forward | 12 | 88.71 |
| CUDA | record_param_comms | 21 | 40.86 |
| CUDA | nccl:all_reduce | 15 | 40.20 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 38.18 |
| CUDA | aten::copy_ | 2262 | 11.88 |
| CUDA | aten::mm | 612 | 7.73 |
| CUDA | Optimizer.step#AdamW.step | 3 | 5.99 |
| CPU | ProfilerStep* | 3 | 192.89 |
| CPU | cudaLaunchKernel | 5997 | 41.64 |
| CPU | DistributedDataParallel.forward | 12 | 20.77 |
| CPU | aten::copy_ | 2262 | 13.42 |
| CPU | aten::empty_strided | 2259 | 12.31 |
| CPU | aten::mm | 612 | 11.71 |
| CPU | aten::addmm | 288 | 10.62 |
| CPU | cudaStreamSynchronize | 18 | 10.12 |
| Collective | nccl:all_reduce | 15 | 40.20 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 38.18 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.02 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.66 |
| Collective | nccl:broadcast | 3 | 0.66 |
| Collective | c10d::broadcast_ | 3 | 0.07 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。


## fsdp_8gpu

# MiniTrainBench Profiler 摘要

- Strategy：fsdp
- GPU 数：8
- 精度：bf16
- Trace 目录：`/workspace/results/profile_8gpu/fsdp_8gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.20 | 0.01 | 0.19 | 0.22 |
| forward_backward_ms | 184.84 | 1.72 | 182.54 | 186.68 |
| optimizer_step_ms | 4.87 | 0.22 | 4.71 | 5.18 |
| step_time_ms | 189.97 | 1.79 | 187.50 | 191.67 |
| tokens_per_sec | 43125.32 | 408.94 | 42740.68 | 43691.66 |
| grad_norm | 103.58 | 31.82 | 74.15 | 147.78 |
| learning_rate | 0.00 | 0.00 | 0.00 | 0.00 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 189.79 |
| Step p50 (ms) | 189.99 |
| Step max (ms) | 190.18 |
| Straggler ratio (max/p50) | 1.001 |
| 每 rank collective total (ms, mean) | 313.61 |
| 每 rank collective/step (ms, mean) | 104.54 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 567.58 |
| CUDA | record_param_comms | 246 | 406.02 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 205.95 |
| CUDA | nccl:_all_gather_base | 156 | 205.95 |
| CUDA | FullyShardedDataParallel.forward | 84 | 203.11 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 197.18 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 191.42 |
| CUDA | nccl:_reduce_scatter_base | 84 | 191.42 |
| CPU | ProfilerStep* | 3 | 332.69 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 42.92 |
| CPU | cudaLaunchKernel | 5262 | 42.84 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 39.11 |
| CPU | cudaStreamSynchronize | 18 | 37.74 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 36.10 |
| CPU | FullyShardedDataParallel.forward | 84 | 26.14 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 21.75 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 205.95 |
| Collective | nccl:_all_gather_base | 156 | 205.95 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 191.42 |
| Collective | nccl:_reduce_scatter_base | 84 | 191.42 |
| Collective | nccl:all_reduce | 6 | 8.65 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 5.69 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.96 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.31 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 567.25 |
| CUDA | record_param_comms | 246 | 418.90 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 209.72 |
| CUDA | nccl:_all_gather_base | 156 | 209.72 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 206.89 |
| CUDA | FullyShardedDataParallel.forward | 84 | 202.70 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 201.44 |
| CUDA | nccl:_reduce_scatter_base | 84 | 201.44 |
| CPU | ProfilerStep* | 3 | 328.43 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 43.17 |
| CPU | cudaLaunchKernel | 5334 | 41.76 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 41.15 |
| CPU | cudaStreamSynchronize | 18 | 38.39 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 35.67 |
| CPU | FullyShardedDataParallel.forward | 84 | 25.99 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 23.79 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 209.72 |
| Collective | nccl:_all_gather_base | 156 | 209.72 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 201.44 |
| Collective | nccl:_reduce_scatter_base | 84 | 201.44 |
| Collective | nccl:all_reduce | 6 | 7.74 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 4.94 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.80 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.32 |

### Rank 2

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 566.52 |
| CUDA | record_param_comms | 246 | 324.51 |
| CUDA | FullyShardedDataParallel.forward | 84 | 222.50 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 169.86 |
| CUDA | nccl:_all_gather_base | 156 | 169.86 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 154.46 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 148.08 |
| CUDA | nccl:_reduce_scatter_base | 84 | 148.08 |
| CPU | ProfilerStep* | 3 | 336.09 |
| CPU | cudaLaunchKernel | 5334 | 45.58 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 43.64 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 41.25 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 36.73 |
| CPU | cudaStreamSynchronize | 18 | 26.93 |
| CPU | FullyShardedDataParallel.forward | 84 | 25.89 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 23.97 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 169.86 |
| Collective | nccl:_all_gather_base | 156 | 169.86 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 148.08 |
| Collective | nccl:_reduce_scatter_base | 84 | 148.08 |
| Collective | nccl:all_reduce | 6 | 6.57 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 3.80 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.77 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.32 |

### Rank 3

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 567.75 |
| CUDA | record_param_comms | 246 | 390.22 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 204.19 |
| CUDA | nccl:_all_gather_base | 156 | 204.19 |
| CUDA | FullyShardedDataParallel.forward | 84 | 203.94 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 183.27 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 177.17 |
| CUDA | nccl:_reduce_scatter_base | 84 | 177.17 |
| CPU | ProfilerStep* | 3 | 333.83 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 43.69 |
| CPU | cudaLaunchKernel | 5262 | 40.78 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 40.06 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 36.66 |
| CPU | cudaStreamSynchronize | 18 | 35.89 |
| CPU | FullyShardedDataParallel.forward | 84 | 26.18 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 22.22 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 204.19 |
| Collective | nccl:_all_gather_base | 156 | 204.19 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 177.17 |
| Collective | nccl:_reduce_scatter_base | 84 | 177.17 |
| Collective | nccl:all_reduce | 6 | 8.86 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 5.88 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.98 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.34 |

### Rank 4

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 566.43 |
| CUDA | FullyShardedDataParallel.forward | 84 | 268.89 |
| CUDA | record_param_comms | 246 | 89.81 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 53.97 |
| CUDA | nccl:_all_gather_base | 156 | 53.97 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 38.21 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 28.37 |
| CUDA | nccl:_reduce_scatter_base | 84 | 28.37 |
| CPU | ProfilerStep* | 3 | 344.25 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 45.85 |
| CPU | cudaLaunchKernel | 5262 | 43.80 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 40.68 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 37.91 |
| CPU | FullyShardedDataParallel.forward | 84 | 30.32 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 24.24 |
| CPU | record_param_comms | 246 | 14.28 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 53.97 |
| Collective | nccl:_all_gather_base | 156 | 53.97 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 28.37 |
| Collective | nccl:_reduce_scatter_base | 84 | 28.37 |
| Collective | nccl:all_reduce | 6 | 7.46 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 4.56 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.90 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.35 |

### Rank 5

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 567.14 |
| CUDA | record_param_comms | 246 | 358.33 |
| CUDA | FullyShardedDataParallel.forward | 84 | 216.94 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 184.85 |
| CUDA | nccl:_all_gather_base | 156 | 184.85 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 172.21 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 165.75 |
| CUDA | nccl:_reduce_scatter_base | 84 | 165.75 |
| CPU | ProfilerStep* | 3 | 331.40 |
| CPU | cudaLaunchKernel | 5334 | 43.36 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 43.25 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 41.87 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 36.27 |
| CPU | cudaStreamSynchronize | 18 | 32.41 |
| CPU | FullyShardedDataParallel.forward | 84 | 26.67 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 24.18 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 184.85 |
| Collective | nccl:_all_gather_base | 156 | 184.85 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 165.75 |
| Collective | nccl:_reduce_scatter_base | 84 | 165.75 |
| Collective | nccl:all_reduce | 6 | 7.73 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 4.94 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.79 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.31 |

### Rank 6

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 567.30 |
| CUDA | record_param_comms | 246 | 351.08 |
| CUDA | FullyShardedDataParallel.forward | 84 | 210.25 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 192.29 |
| CUDA | nccl:_all_gather_base | 156 | 192.29 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 156.27 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 150.16 |
| CUDA | nccl:_reduce_scatter_base | 84 | 150.16 |
| CPU | ProfilerStep* | 3 | 339.07 |
| CPU | cudaLaunchKernel | 5262 | 44.09 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 43.75 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 39.93 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 36.81 |
| CPU | cudaStreamSynchronize | 18 | 28.20 |
| CPU | FullyShardedDataParallel.forward | 84 | 26.60 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 21.96 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 192.29 |
| Collective | nccl:_all_gather_base | 156 | 192.29 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 150.16 |
| Collective | nccl:_reduce_scatter_base | 84 | 150.16 |
| Collective | nccl:all_reduce | 6 | 8.62 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 5.67 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.95 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.32 |

### Rank 7

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 564.54 |
| CUDA | FullyShardedDataParallel.forward | 84 | 250.43 |
| CUDA | record_param_comms | 246 | 169.99 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 93.87 |
| CUDA | nccl:_all_gather_base | 156 | 93.87 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 84 | 83.83 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 75.99 |
| CUDA | nccl:_reduce_scatter_base | 84 | 75.99 |
| CPU | ProfilerStep* | 3 | 342.85 |
| CPU | FullyShardedDataParallel._pre_forward | 84 | 45.76 |
| CPU | FullyShardedDataParallel._post_backward_hook | 84 | 44.79 |
| CPU | cudaLaunchKernel | 5460 | 44.24 |
| CPU | FullyShardedDataParallel._pre_backward_prefetch | 84 | 38.08 |
| CPU | FullyShardedDataParallel._post_forward | 84 | 27.37 |
| CPU | FullyShardedDataParallel.forward | 84 | 27.02 |
| CPU | record_param_comms | 246 | 14.61 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 156 | 93.87 |
| Collective | nccl:_all_gather_base | 156 | 93.87 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 84 | 75.99 |
| Collective | nccl:_reduce_scatter_base | 84 | 75.99 |
| Collective | c10d::_reduce_scatter_base_ | 84 | 1.32 |
| Collective | nccl:all_reduce | 6 | 0.14 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.08 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.05 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
