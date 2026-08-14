## 生成的 Benchmark 结果

### 实验环境

| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | Image ID | Base image | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.10.0+cu130 | 13.0 | 91501 | 2.28.9 | 580.173.02 | NVIDIA A100-SXM4-40GB | 048ca3ed54df | sha256:95dcdfd0b564 | pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b | 完整 |

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 | 11.96 | 3.09 | 33609.95 | 15.23 | 481.47 | 100.00% | - | - | 1 |
| ddp | 2 | bf16 | 0.04 | 14.97 | 3.55 | 54956.81 | 18.63 | 567.13 | 81.76% | - | - | 1 |
| ddp | 4 | bf16 | 0.04 | 14.35 | 3.34 | 114609.10 | 17.87 | 567.13 | 85.25% | - | - | 1 |
| ddp | 8 | bf16 | 0.05 | 14.44 | 3.40 | 227441.15 | 18.01 | 567.13 | 84.59% | - | - | 1 |
| fsdp | 1 | bf16 | 0.07 | 30.22 | 4.53 | 14647.57 | 34.95 | 479.77 | 100.00% | 0.35% | 19.72 | 1 |
| fsdp | 2 | bf16 | 0.05 | 29.30 | 3.41 | 31187.29 | 32.83 | 276.23 | 106.46% | 51.29% | 14.20 | 1 |
| fsdp | 4 | bf16 | 0.05 | 28.98 | 2.89 | 64193.75 | 31.90 | 208.09 | 109.56% | 63.31% | 14.03 | 1 |
| fsdp | 8 | bf16 | 0.05 | 29.57 | 2.82 | 126662.75 | 32.34 | 174.59 | 108.09% | 69.22% | 14.33 | 1 |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| ddp | 1 | DDPStrategy | 否 | 7 | 3584 | single_run | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 2 | DDPStrategy | 否 | 7 | 7168 | single_run | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 4 | DDPStrategy | 否 | 7 | 14336 | single_run | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 8 | DDPStrategy | 否 | 7 | 28672 | single_run | auto | last | 1 | - | - | 3 | 0 | - | - |
| fsdp | 1 | FSDPStrategy | 否 | 7 | 3584 | single_run | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 7 | 7168 | single_run | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 4 | FSDPStrategy | 否 | 7 | 14336 | single_run | auto | every | 1 | - | - | 3 | 0 | - | - |
| fsdp | 8 | FSDPStrategy | 否 | 7 | 28672 | single_run | auto | every | 1 | - | - | 3 | 0 | - | - |

#### 稳定性指标

| 策略 | GPU 数 | LR scheduler | 当前 LR | Grad norm mean | Grad norm max | 裁剪阈值 | 裁剪步数 | 非有限值策略 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ddp | 1 | constant | 0.0003 | 101.46 | 164.154 | 0 | 0 | all_rank_fail_fast |
| ddp | 2 | constant | 0.0003 | 90.8456 | 158.486 | 0 | 0 | all_rank_fail_fast |
| ddp | 4 | constant | 0.0003 | 95.9837 | 154.218 | 0 | 0 | all_rank_fail_fast |
| ddp | 8 | constant | 0.0003 | 98.2008 | 150.756 | 0 | 0 | all_rank_fail_fast |
| fsdp | 1 | constant | 0.0003 | 101.471 | 164.19 | 0 | 0 | all_rank_fail_fast |
| fsdp | 2 | constant | 0.0003 | 105.756 | 158.063 | 0 | 0 | all_rank_fail_fast |
| fsdp | 4 | constant | 0.0003 | 91.3279 | 153.213 | 0 | 0 | all_rank_fail_fast |
| fsdp | 8 | constant | 0.0003 | 86.3853 | 149.828 | 0 | 0 | all_rank_fail_fast |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| all_reduce | 8 | - | 1024 | 0.167 | 0.025 | ok |
| all_gather | 8 | - | 1024 | 0.234 | 0.140 | ok |
| reduce_scatter | 8 | - | 1024 | 0.075 | 0.438 | ok |
| all_to_all | 8 | equal | 1024 | 0.062 | 0.532 | ok |
| all_to_all | 8 | uneven | 1024 | 0.058 | 0.570 | ok |
| all_reduce | 8 | - | 1048576 | 0.116 | 36.145 | ok |
| all_gather | 8 | - | 1048576 | 0.334 | 100.442 | ok |
| reduce_scatter | 8 | - | 1048576 | 0.243 | 138.101 | ok |
| all_to_all | 8 | equal | 1048576 | 0.213 | 157.890 | ok |
| all_to_all | 8 | uneven | 1048576 | 0.501 | 67.014 | ok |
| all_reduce | 8 | - | 16777216 | 0.719 | 93.297 | ok |
| all_gather | 8 | - | 16777216 | 3.253 | 165.030 | ok |
| reduce_scatter | 8 | - | 16777216 | 2.219 | 241.894 | ok |
| all_to_all | 8 | equal | 16777216 | 2.415 | 222.320 | ok |
| all_to_all | 8 | uneven | 16777216 | 6.866 | 78.187 | ok |

小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。all-to-all 对应 MoE expert parallel 的 token dispatch/combine，可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。
