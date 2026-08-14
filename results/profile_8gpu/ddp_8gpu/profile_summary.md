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
