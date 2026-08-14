# MiniTrainBench Profiler 摘要

- Strategy：ddp
- GPU 数：2
- 精度：bf16
- Trace 目录：`/workspace/results/profile/ddp_2gpu`

## Step 拆分

| 指标 | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| data_time_ms | 0.21 | 0.01 | 0.20 | 0.22 |
| forward_backward_ms | 94.11 | 0.09 | 94.00 | 94.22 |
| optimizer_step_ms | 5.48 | 0.01 | 5.47 | 5.50 |
| step_time_ms | 99.98 | 0.06 | 99.89 | 100.05 |
| tokens_per_sec | 20484.25 | 13.05 | 20470.17 | 20501.61 |
| grad_norm | 105.73 | 37.30 | 61.65 | 152.86 |
| learning_rate | 0.00 | 0.00 | 0.00 | 0.00 |

## Rank 诊断

| 指标 | 值 |
| --- | ---: |
| Step min (ms) | 99.87 |
| Step p50 (ms) | 99.93 |
| Step max (ms) | 99.98 |
| Straggler ratio (max/p50) | 1.001 |
| 每 rank collective total (ms, mean) | 19.55 |
| 每 rank collective/step (ms, mean) | 6.52 |

计算通信 overlap：未确定。`key_averages()` 不保留跨 CUDA stream 的时间关系；请以每 rank Chrome trace 的实际时间线作为证据。

## Rank Top Ops

### Rank 0

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 291.63 |
| CUDA | DistributedDataParallel.forward | 12 | 84.98 |
| CUDA | record_param_comms | 21 | 36.49 |
| CUDA | nccl:all_reduce | 15 | 36.46 |
| CUDA | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 33.07 |
| CUDA | aten::copy_ | 2262 | 9.86 |
| CUDA | aten::mm | 612 | 6.26 |
| CUDA | Optimizer.step#AdamW.step | 3 | 5.95 |
| CPU | ProfilerStep* | 3 | 189.54 |
| CPU | cudaLaunchKernel | 5997 | 43.96 |
| CPU | DistributedDataParallel.forward | 12 | 20.92 |
| CPU | aten::copy_ | 2262 | 13.17 |
| CPU | aten::empty_strided | 2259 | 11.97 |
| CPU | aten::mm | 612 | 11.65 |
| CPU | aten::empty | 1839 | 9.15 |
| CPU | cudaStreamSynchronize | 18 | 9.09 |
| Collective | nccl:all_reduce | 15 | 36.46 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 33.07 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 3.38 |
| Collective | c10d::broadcast_ | 3 | 0.07 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.03 |
| Collective | nccl:broadcast | 3 | 0.03 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

### Rank 1

| 类型 | Name | Calls | Time (ms) |
| --- | --- | ---: | ---: |
| CUDA | ProfilerStep* | 3 | 291.27 |
| CUDA | DistributedDataParallel.forward | 12 | 87.21 |
| CUDA | aten::copy_ | 2262 | 11.80 |
| CUDA | aten::mm | 612 | 7.60 |
| CUDA | Optimizer.step#AdamW.step | 3 | 6.06 |
| CUDA | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 756 | 5.31 |
| CUDA | aten::add_ | 1212 | 4.43 |
| CUDA | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 852 | 3.97 |
| CPU | ProfilerStep* | 3 | 195.37 |
| CPU | cudaLaunchKernel | 5997 | 44.34 |
| CPU | DistributedDataParallel.forward | 12 | 20.41 |
| CPU | aten::copy_ | 2262 | 13.31 |
| CPU | aten::empty_strided | 2259 | 12.31 |
| CPU | aten::mm | 612 | 11.83 |
| CPU | aten::empty | 1839 | 9.28 |
| CPU | aten::addmm | 288 | 8.53 |
| Collective | nccl:all_reduce | 15 | 2.56 |
| Collective | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 12 | 2.51 |
| Collective | c10d::broadcast_ | 3 | 0.08 |
| Collective | ncclDevKernel_AllReduce_Sum_u32_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.05 |
| Collective | ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>) | 3 | 0.04 |
| Collective | nccl:broadcast | 3 | 0.04 |
| Collective | nccl:broadcast | 3 | 0.00 |
| Collective | nccl:all_reduce | 15 | 0.00 |

原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 `chrome://tracing` 或 Perfetto 打开 rank trace。
