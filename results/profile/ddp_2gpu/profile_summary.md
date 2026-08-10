# MiniTrainBench Profiler 摘要

- Strategy：ddp
- GPU 数：2
- 精度：bf16
- Trace 目录：`/workspace/results/profile/ddp_2gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.10 | 0.02 | 0.07 | 0.12 |
| forward_backward_ms | 27.27 | 0.85 | 26.41 | 28.43 |
| optimizer_step_ms | 3.14 | 0.18 | 3.00 | 3.39 |
| step_time_ms | 32.22 | 3.07 | 29.83 | 36.56 |
| tokens_per_sec | 16028.80 | 1433.91 | 14006.22 | 17166.64 |

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 86.75 |
| CUDA | DistributedDataParallel.forward | 3 | 23.56 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.45 |
| CUDA | aten::copy_ | 738 | 4.03 |
| CUDA | record_param_comms | 18 | 2.69 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 2.65 |
| CUDA | nccl:all_reduce | 12 | 2.65 |
| CUDA | aten::mm | 153 | 1.92 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 2.65 |
| Collective | nccl:all_reduce | 12 | 2.65 |
| Collective | c10d::broadcast_ | 3 | 0.12 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.04 |
| Collective | nccl:broadcast | 3 | 0.04 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 86.07 |
| CUDA | record_param_comms | 18 | 34.84 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 28.28 |
| CUDA | nccl:all_reduce | 12 | 28.28 |
| CUDA | DistributedDataParallel.forward | 3 | 16.68 |
| CUDA | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 6.56 |
| CUDA | nccl:broadcast | 3 | 6.56 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.49 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 28.28 |
| Collective | nccl:all_reduce | 12 | 28.28 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 6.56 |
| Collective | nccl:broadcast | 3 | 6.56 |
| Collective | c10d::broadcast_ | 3 | 0.13 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
