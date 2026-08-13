# MiniTrainBench Profiler 摘要

- Strategy：ddp
- GPU 数：8
- 精度：bf16
- Trace 目录：`/workspace/results/profile_8gpu/ddp_8gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.20 | 0.02 | 0.18 | 0.23 |
| forward_backward_ms | 95.87 | 0.12 | 95.76 | 96.04 |
| optimizer_step_ms | 2.84 | 0.06 | 2.77 | 2.91 |
| step_time_ms | 99.11 | 0.18 | 98.94 | 99.35 |
| tokens_per_sec | 82659.27 | 147.79 | 82455.34 | 82800.88 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 99.09 |
| Step p50 (ms) | 99.13 |
| Step max (ms) | 99.31 |
| Straggler ratio (max/p50) | 1.002 |
| 每 rank collective total (ms, mean) | 22.00 |
| 每 rank collective/step (ms, mean) | 7.33 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.41 |
| CUDA | DistributedDataParallel.forward | 12 | 88.51 |
| CUDA | record_param_comms | 18 | 26.76 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 26.71 |
| CUDA | nccl:all_reduce | 12 | 26.71 |
| CUDA | aten::copy_ | 2259 | 11.91 |
| CUDA | aten::mm | 612 | 7.82 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.14 |
| CPU | ProfilerStep* | 3 | 191.86 |
| CPU | cudaLaunchKernel | 5880 | 43.28 |
| CPU | DistributedDataParallel.forward | 12 | 21.53 |
| CPU | aten::copy_ | 2259 | 13.61 |
| CPU | aten::empty_strided | 2256 | 12.33 |
| CPU | aten::mm | 612 | 12.11 |
| CPU | aten::empty | 1815 | 9.18 |
| CPU | aten::addmm | 288 | 8.65 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 26.71 |
| Collective | nccl:all_reduce | 12 | 26.71 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.05 |
| Collective | nccl:broadcast | 3 | 0.05 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 287.96 |
| CUDA | DistributedDataParallel.forward | 12 | 89.23 |
| CUDA | record_param_comms | 18 | 21.62 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 21.56 |
| CUDA | nccl:all_reduce | 12 | 21.56 |
| CUDA | aten::copy_ | 2259 | 11.89 |
| CUDA | aten::mm | 612 | 7.68 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.27 |
| CPU | ProfilerStep* | 3 | 191.88 |
| CPU | cudaLaunchKernel | 5880 | 42.14 |
| CPU | DistributedDataParallel.forward | 12 | 22.27 |
| CPU | aten::copy_ | 2259 | 13.61 |
| CPU | aten::empty_strided | 2256 | 12.32 |
| CPU | aten::mm | 612 | 11.93 |
| CPU | aten::empty | 1815 | 9.17 |
| CPU | aten::addmm | 288 | 8.72 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 21.56 |
| Collective | nccl:all_reduce | 12 | 21.56 |
| Collective | c10d::broadcast_ | 3 | 0.10 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.06 |
| Collective | nccl:broadcast | 3 | 0.06 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 2

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.40 |
| CUDA | DistributedDataParallel.forward | 12 | 88.38 |
| CUDA | record_param_comms | 18 | 25.02 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 24.35 |
| CUDA | nccl:all_reduce | 12 | 24.35 |
| CUDA | aten::copy_ | 2259 | 11.94 |
| CUDA | aten::mm | 612 | 7.67 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.18 |
| CPU | ProfilerStep* | 3 | 192.80 |
| CPU | cudaLaunchKernel | 5880 | 43.62 |
| CPU | DistributedDataParallel.forward | 12 | 21.54 |
| CPU | aten::copy_ | 2259 | 13.54 |
| CPU | aten::empty_strided | 2256 | 12.46 |
| CPU | aten::mm | 612 | 12.00 |
| CPU | aten::empty | 1815 | 9.18 |
| CPU | aten::addmm | 288 | 8.62 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 24.35 |
| Collective | nccl:all_reduce | 12 | 24.35 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.67 |
| Collective | nccl:broadcast | 3 | 0.67 |
| Collective | c10d::broadcast_ | 3 | 0.09 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 3

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.55 |
| CUDA | DistributedDataParallel.forward | 12 | 87.98 |
| CUDA | record_param_comms | 18 | 30.59 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 29.70 |
| CUDA | nccl:all_reduce | 12 | 29.70 |
| CUDA | aten::copy_ | 2259 | 11.77 |
| CUDA | aten::mm | 612 | 7.74 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.19 |
| CPU | ProfilerStep* | 3 | 191.68 |
| CPU | cudaLaunchKernel | 5880 | 41.52 |
| CPU | DistributedDataParallel.forward | 12 | 21.73 |
| CPU | aten::copy_ | 2259 | 13.54 |
| CPU | aten::empty_strided | 2256 | 12.32 |
| CPU | aten::mm | 612 | 11.98 |
| CPU | aten::empty | 1815 | 9.12 |
| CPU | aten::addmm | 288 | 8.85 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 29.70 |
| Collective | nccl:all_reduce | 12 | 29.70 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.90 |
| Collective | nccl:broadcast | 3 | 0.90 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 4

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.83 |
| CUDA | DistributedDataParallel.forward | 12 | 89.85 |
| CUDA | record_param_comms | 18 | 12.18 |
| CUDA | aten::copy_ | 2259 | 11.88 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 11.67 |
| CUDA | nccl:all_reduce | 12 | 11.67 |
| CUDA | aten::mm | 612 | 7.62 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.27 |
| CPU | ProfilerStep* | 3 | 194.64 |
| CPU | cudaLaunchKernel | 5880 | 44.58 |
| CPU | DistributedDataParallel.forward | 12 | 21.94 |
| CPU | aten::copy_ | 2259 | 13.67 |
| CPU | aten::empty_strided | 2256 | 12.36 |
| CPU | aten::mm | 612 | 12.01 |
| CPU | aten::empty | 1815 | 9.19 |
| CPU | aten::addmm | 288 | 8.67 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 11.67 |
| Collective | nccl:all_reduce | 12 | 11.67 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.51 |
| Collective | nccl:broadcast | 3 | 0.51 |
| Collective | c10d::broadcast_ | 3 | 0.10 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 5

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.60 |
| CUDA | DistributedDataParallel.forward | 12 | 89.75 |
| CUDA | aten::copy_ | 2259 | 12.21 |
| CUDA | aten::mm | 612 | 7.62 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.22 |
| CUDA | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 756 | 5.35 |
| CUDA | aten::add_ | 1212 | 4.53 |
| CUDA | record_param_comms | 18 | 4.47 |
| CPU | ProfilerStep* | 3 | 194.56 |
| CPU | cudaLaunchKernel | 5880 | 44.36 |
| CPU | DistributedDataParallel.forward | 12 | 21.71 |
| CPU | aten::copy_ | 2259 | 13.69 |
| CPU | aten::empty_strided | 2256 | 12.36 |
| CPU | aten::mm | 612 | 12.01 |
| CPU | aten::empty | 1815 | 9.19 |
| CPU | aten::addmm | 288 | 8.87 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 4.42 |
| Collective | nccl:all_reduce | 12 | 4.42 |
| Collective | c10d::broadcast_ | 3 | 0.11 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.06 |
| Collective | nccl:broadcast | 3 | 0.06 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 6

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.35 |
| CUDA | DistributedDataParallel.forward | 12 | 87.51 |
| CUDA | record_param_comms | 18 | 25.85 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 23.63 |
| CUDA | nccl:all_reduce | 12 | 23.63 |
| CUDA | aten::copy_ | 2259 | 11.80 |
| CUDA | aten::mm | 612 | 7.75 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.28 |
| CPU | ProfilerStep* | 3 | 192.19 |
| CPU | cudaLaunchKernel | 5880 | 43.08 |
| CPU | DistributedDataParallel.forward | 12 | 21.79 |
| CPU | aten::copy_ | 2259 | 13.73 |
| CPU | aten::empty_strided | 2256 | 12.35 |
| CPU | aten::mm | 612 | 11.92 |
| CPU | aten::empty | 1815 | 9.43 |
| CPU | aten::addmm | 288 | 8.77 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 23.63 |
| Collective | nccl:all_reduce | 12 | 23.63 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.22 |
| Collective | nccl:broadcast | 3 | 2.22 |
| Collective | c10d::broadcast_ | 3 | 0.10 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

### Rank 7

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 288.42 |
| CUDA | DistributedDataParallel.forward | 12 | 86.97 |
| CUDA | record_param_comms | 18 | 29.49 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 27.19 |
| CUDA | nccl:all_reduce | 12 | 27.19 |
| CUDA | aten::copy_ | 2259 | 11.78 |
| CUDA | aten::mm | 612 | 7.71 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.26 |
| CPU | ProfilerStep* | 3 | 191.72 |
| CPU | cudaLaunchKernel | 5880 | 42.45 |
| CPU | DistributedDataParallel.forward | 12 | 21.78 |
| CPU | aten::copy_ | 2259 | 13.73 |
| CPU | aten::empty_strided | 2256 | 12.34 |
| CPU | aten::mm | 612 | 12.02 |
| CPU | aten::empty | 1815 | 9.18 |
| CPU | aten::addmm | 288 | 8.68 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 27.19 |
| Collective | nccl:all_reduce | 12 | 27.19 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 2.30 |
| Collective | nccl:broadcast | 3 | 2.30 |
| Collective | c10d::broadcast_ | 3 | 0.10 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 12 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
