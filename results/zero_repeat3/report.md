## 生成的 Benchmark 结果

### 实验环境

| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | Image ID | Base image | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.10.0+cu130 | 13.0 | 91501 | 2.28.9 | 580.173.02 | NVIDIA A100-SXM4-40GB | 048ca3ed54df | sha256:9dfe0ef5c91f | pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b | 完整 |

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 ± 0.00 | 11.97 ± 0.04 | 3.05 ± 0.01 | 33649.64 ± 79.52 | 15.22 ± 0.04 | 481.06 ± 0.29 | 100.00% | - | - | 3 |
| ddp | 2 | bf16 | 0.04 ± 0.00 | 13.75 ± 0.04 | 3.14 ± 0.02 | 60101.10 ± 126.15 | 17.04 ± 0.04 | 616.18 ± 38.80 | 89.30% | - | - | 3 |
| ddp | 4 | bf16 | 0.04 ± 0.00 | 14.64 ± 0.05 | 3.52 ± 0.04 | 111954.05 ± 328.15 | 18.29 ± 0.05 | 630.13 ± 44.55 | 83.18% | - | - | 3 |
| ddp | 8 | bf16 | 0.05 ± 0.01 | 14.74 ± 0.11 | 3.46 ± 0.05 | 223572.89 ± 1665.11 | 18.32 ± 0.14 | 630.13 ± 44.55 | 83.05% | - | - | 3 |
| deepspeed_zero2 | 1 | bf16 | 0.04 ± 0.00 | 19.25 ± 0.33 | 12.39 ± 0.05 | 16040.24 ± 144.71 | 31.92 ± 0.29 | 2616.74 ± 253.04 | 100.00% | -443.96% | 16.71 | 3 |
| deepspeed_zero2 | 2 | bf16 | 0.09 ± 0.07 | 17.75 ± 0.13 | 7.93 ± 0.16 | 39698.76 ± 218.98 | 25.80 ± 0.14 | 2350.75 ± 144.62 | 123.75% | -281.50% | 8.76 | 3 |
| deepspeed_zero2 | 4 | bf16 | 0.04 ± 0.00 | 18.21 ± 0.44 | 5.24 ± 0.06 | 86783.55 ± 1665.64 | 23.61 ± 0.46 | 2217.97 ± 90.38 | 135.26% | -251.98% | 5.31 | 3 |
| deepspeed_zero2 | 8 | bf16 | 0.04 ± 0.00 | 17.93 ± 0.19 | 3.41 ± 0.03 | 190638.99 ± 1937.99 | 21.49 ± 0.22 | 2151.80 ± 64.65 | 148.56% | -241.48% | 3.17 | 3 |
| deepspeed_zero3 | 1 | bf16 | 0.04 ± 0.00 | 57.93 ± 0.02 | 9.15 ± 0.18 | 7595.51 ± 15.98 | 67.41 ± 0.14 | 2819.95 ± 1068.54 | 100.00% | -486.20% | 52.19 | 3 |
| deepspeed_zero3 | 2 | bf16 | 0.05 ± 0.00 | 60.78 ± 0.44 | 12.42 ± 0.39 | 14044.27 ± 98.74 | 72.92 ± 0.51 | 2418.26 ± 923.83 | 92.45% | -292.46% | 55.88 | 3 |
| deepspeed_zero3 | 4 | bf16 | 0.05 ± 0.00 | 62.23 ± 0.14 | 12.81 ± 0.05 | 27511.18 ± 21.70 | 74.44 ± 0.06 | 2243.80 ± 852.14 | 90.55% | -256.08% | 56.15 | 3 |
| deepspeed_zero3 | 8 | bf16 | 0.05 ± 0.00 | 65.52 ± 0.81 | 13.15 ± 0.05 | 52530.42 ± 580.95 | 77.98 ± 0.86 | 2157.75 ± 816.70 | 86.45% | -242.43% | 59.66 | 3 |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| ddp | 1 | DDPStrategy | 否 | 25 | 12800 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 2 | DDPStrategy | 否 | 25 | 25600 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 4 | DDPStrategy | 否 | 25 | 51200 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 8 | DDPStrategy | 否 | 25 | 102400 | independent_reinitialize | auto | last | 1 | - | - | 3 | 0 | - | - |
| deepspeed_zero2 | 1 | DeepSpeedZeRO2 | 否 | 25 | 12800 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero2 | 2 | DeepSpeedZeRO2 | 否 | 25 | 25600 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero2 | 4 | DeepSpeedZeRO2 | 否 | 25 | 51200 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero2 | 8 | DeepSpeedZeRO2 | 否 | 25 | 102400 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero3 | 1 | DeepSpeedZeRO3 | 否 | 25 | 12800 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero3 | 2 | DeepSpeedZeRO3 | 否 | 25 | 25600 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero3 | 4 | DeepSpeedZeRO3 | 否 | 25 | 51200 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |
| deepspeed_zero3 | 8 | DeepSpeedZeRO3 | 否 | 25 | 102400 | independent_reinitialize | - | - | - | - | - | 0 | 0 | - | - |

#### 稳定性指标

| 策略 | GPU 数 | LR scheduler | 当前 LR | Grad norm mean | Grad norm max | 裁剪阈值 | 裁剪步数 | 非有限值策略 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ddp | 1 | constant | 0.0003 | 25.3416 | 79.4016 | 0 | 0 | all_rank_fail_fast |
| ddp | 2 | constant | 0.0003 | 22.1767 | 60.2108 | 0 | 0 | all_rank_fail_fast |
| ddp | 4 | constant | 0.0003 | 22.7692 | 76.5752 | 0 | 0 | all_rank_fail_fast |
| ddp | 8 | constant | 0.0003 | 22.9263 | 74.7841 | 0 | 0 | all_rank_fail_fast |
| deepspeed_zero2 | 1 | - | - | - | - | - | - | - |
| deepspeed_zero2 | 2 | - | - | - | - | - | - | - |
| deepspeed_zero2 | 4 | - | - | - | - | - | - | - |
| deepspeed_zero2 | 8 | - | - | - | - | - | - | - |
| deepspeed_zero3 | 1 | - | - | - | - | - | - | - |
| deepspeed_zero3 | 2 | - | - | - | - | - | - | - |
| deepspeed_zero3 | 4 | - | - | - | - | - | - | - |
| deepspeed_zero3 | 8 | - | - | - | - | - | - | - |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
