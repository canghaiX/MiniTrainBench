## 生成的 Benchmark 结果

### 实验环境

| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | Image ID | Base image | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.10.0+cu130 | 13.0 | 91501 | 2.28.9 | 580.173.02 | NVIDIA A100-SXM4-40GB | 048ca3ed54df | sha256:95dcdfd0b564 | pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b | 完整 |

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 ± 0.00 | 12.11 ± 0.16 | 3.10 ± 0.02 | 33247.71 ± 388.25 | 15.40 ± 0.18 | 481.06 ± 0.29 | 100.00% | - | - | 3 |
| ddp | 2 | bf16 | 0.04 ± 0.00 | 14.20 ± 0.09 | 3.37 ± 0.06 | 57995.87 ± 323.61 | 17.66 ± 0.10 | 630.37 ± 43.38 | 87.22% | - | - | 3 |
| ddp | 4 | bf16 | 0.04 ± 0.00 | 14.35 ± 0.03 | 3.37 ± 0.02 | 115074.98 ± 341.88 | 17.80 ± 0.05 | 630.13 ± 44.55 | 86.53% | - | - | 3 |
| ddp | 8 | bf16 | 0.04 ± 0.00 | 14.40 ± 0.07 | 3.47 ± 0.07 | 227644.04 ± 1695.72 | 17.99 ± 0.13 | 630.13 ± 44.55 | 85.59% | - | - | 3 |
| fsdp | 1 | bf16 | 0.04 ± 0.00 | 28.23 ± 0.05 | 3.74 ± 0.03 | 15932.41 ± 43.86 | 32.14 ± 0.09 | 657.21 ± 144.96 | 100.00% | -36.62% | 16.73 | 3 |
| fsdp | 2 | bf16 | 0.04 ± 0.00 | 29.01 ± 0.11 | 3.66 ± 0.04 | 31340.13 ± 190.41 | 32.67 ± 0.20 | 274.90 ± 0.94 | 98.35% | 56.39% | 15.02 | 3 |
| fsdp | 4 | bf16 | 0.04 ± 0.00 | 28.87 ± 0.39 | 2.78 ± 0.04 | 64564.38 ± 744.91 | 31.72 ± 0.36 | 210.51 ± 1.71 | 101.31% | 66.59% | 13.93 | 3 |
| fsdp | 8 | bf16 | 0.04 ± 0.00 | 29.15 ± 0.30 | 2.66 ± 0.05 | 128972.61 ± 1061.57 | 31.76 ± 0.26 | 172.83 ± 1.24 | 101.19% | 72.57% | 13.77 | 3 |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| ddp | 1 | DDPStrategy | 否 | 25 | 12800 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 2 | DDPStrategy | 否 | 25 | 25600 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 4 | DDPStrategy | 否 | 25 | 51200 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 8 | DDPStrategy | 否 | 25 | 102400 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| fsdp | 1 | FSDPStrategy | 否 | 25 | 12800 | independent_reinitialize | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 25 | 25600 | independent_reinitialize | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 4 | FSDPStrategy | 否 | 25 | 51200 | independent_reinitialize | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 8 | FSDPStrategy | 否 | 25 | 102400 | independent_reinitialize | auto | every | 1 | - | - | 3 | 0 | - | - |

#### 稳定性指标

| 策略 | GPU 数 | LR scheduler | 当前 LR | Grad norm mean | Grad norm max | 裁剪阈值 | 裁剪步数 | 非有限值策略 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ddp | 1 | constant | 0.0003 | 25.3416 | 79.4016 | 0 | 0 | all_rank_fail_fast |
| ddp | 2 | constant | 0.0003 | 22.1767 | 60.2108 | 0 | 0 | all_rank_fail_fast |
| ddp | 4 | constant | 0.0003 | 22.7692 | 76.5752 | 0 | 0 | all_rank_fail_fast |
| ddp | 8 | constant | 0.0003 | 22.9263 | 74.7841 | 0 | 0 | all_rank_fail_fast |
| fsdp | 1 | constant | 0.0003 | 25.2884 | 79.3228 | 0 | 0 | all_rank_fail_fast |
| fsdp | 2 | constant | 0.0003 | 25.6231 | 83.8429 | 0 | 0 | all_rank_fail_fast |
| fsdp | 4 | constant | 0.0003 | 25.2273 | 64.3188 | 0 | 0 | all_rank_fail_fast |
| fsdp | 8 | constant | 0.0003 | 23.9326 | 72.8899 | 0 | 0 | all_rank_fail_fast |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| all_reduce | 8 | - | 1024 | 0.052 | 0.079 | ok |
| all_gather | 8 | - | 1024 | 0.198 | 0.165 | ok |
| reduce_scatter | 8 | - | 1024 | 0.086 | 0.381 | ok |
| all_to_all | 8 | equal | 1024 | 0.068 | 0.479 | ok |
| all_to_all | 8 | uneven | 1024 | 0.056 | 0.584 | ok |
| all_reduce | 8 | - | 1048576 | 0.115 | 36.379 | ok |
| all_gather | 8 | - | 1048576 | 0.333 | 100.621 | ok |
| reduce_scatter | 8 | - | 1048576 | 0.243 | 138.039 | ok |
| all_to_all | 8 | equal | 1048576 | 0.213 | 157.902 | ok |
| all_to_all | 8 | uneven | 1048576 | 0.489 | 68.583 | ok |
| all_reduce | 8 | - | 16777216 | 0.720 | 93.228 | ok |
| all_gather | 8 | - | 16777216 | 3.348 | 160.373 | ok |
| reduce_scatter | 8 | - | 16777216 | 2.218 | 242.065 | ok |
| all_to_all | 8 | equal | 16777216 | 2.420 | 221.814 | ok |
| all_to_all | 8 | uneven | 16777216 | 6.878 | 78.052 | ok |

小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。all-to-all 对应 MoE expert parallel 的 token dispatch/combine，可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。
