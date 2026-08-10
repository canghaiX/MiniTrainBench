# MiniTrainBench Profiler 摘要

- Strategy：fsdp
- GPU 数：2
- 精度：bf16
- Trace 目录：`/workspace/results/profile/fsdp_2gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.08 | 0.02 | 0.07 | 0.11 |
| forward_backward_ms | 45.92 | 0.53 | 45.20 | 46.45 |
| optimizer_step_ms | 1.62 | 0.03 | 1.59 | 1.66 |
| step_time_ms | 47.71 | 0.53 | 46.97 | 48.19 |
| tokens_per_sec | 10732.93 | 120.09 | 10624.93 | 10900.44 |

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 137.84 |
| CUDA | FullyShardedDataParallel.forward | 21 | 61.17 |
| CUDA | record_param_comms | 60 | 31.25 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 39 | 22.07 |
| CUDA | nccl:_all_gather_base | 39 | 22.07 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 21 | 11.13 |
| CUDA | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 21 | 9.17 |
| CUDA | nccl:_reduce_scatter_base | 21 | 9.17 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 39 | 22.07 |
| Collective | nccl:_all_gather_base | 39 | 22.07 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 21 | 9.17 |
| Collective | nccl:_reduce_scatter_base | 21 | 9.17 |
| Collective | c10d::_reduce_scatter_base_ | 21 | 0.35 |
| Collective | nccl:_all_gather_base | 39 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 21 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 137.31 |
| CUDA | FullyShardedDataParallel.forward | 21 | 65.70 |
| CUDA | record_param_comms | 60 | 5.25 |
| CUDA | Optimizer.step#AdamW.step | 3 | 4.39 |
| CUDA | FullyShardedDataParallel._post_backward_hook | 21 | 3.89 |
| CUDA | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 39 | 3.32 |
| CUDA | nccl:_all_gather_base | 39 | 3.32 |
| CUDA | aten::copy_ | 471 | 2.36 |
| Collective | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 39 | 3.32 |
| Collective | nccl:_all_gather_base | 39 | 3.32 |
| Collective | ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 21 | 1.93 |
| Collective | nccl:_reduce_scatter_base | 21 | 1.93 |
| Collective | c10d::_reduce_scatter_base_ | 21 | 0.33 |
| Collective | nccl:_all_gather_base | 39 | 0.00 |
| Collective | nccl:_reduce_scatter_base | 21 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
