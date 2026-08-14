## 生成的 Benchmark 结果

### 实验环境

| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | Image ID | Base image | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.10.0+cu130 | 13.0 | 91501 | 2.28.9 | 580.173.02 | NVIDIA A100-SXM4-40GB | 048ca3ed54df | sha256:95dcdfd0b564 | pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b | 完整 |

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 2 | bf16 | 0.10 | 51.10 | 3.56 | 37433.80 | 54.71 | 569.65 | - | - | - | 1 |
| ddp | 2 | bf16 | 0.10 | 57.59 | 3.61 | 33399.18 | 61.32 | 570.45 | - | - | - | 1 |
| fsdp | 2 | bf16 | 0.11 | 116.55 | 4.31 | 16973.97 | 120.66 | 267.21 | - | 53.16% | 59.34 | 1 |
| fsdp | 2 | bf16 | 0.10 | 113.20 | 3.33 | 17578.88 | 116.50 | 267.21 | - | 53.16% | 55.18 | 1 |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| ddp | 2 | DDPStrategy | 否 | 7 | 14336 | single_run | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 2 | DDPStrategy | 否 | 7 | 14336 | single_run | every | every | 4 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 7 | 14336 | single_run | auto | every | 4 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 7 | 14336 | single_run | last | last | 1 | - | - | 3 | 0 | - | - |

#### 稳定性指标

| 策略 | GPU 数 | LR scheduler | 当前 LR | Grad norm mean | Grad norm max | 裁剪阈值 | 裁剪步数 | 非有限值策略 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ddp | 2 | constant | 0.0003 | 82.5295 | 152.86 | 0 | 0 | all_rank_fail_fast |
| ddp | 2 | constant | 0.0003 | 82.4914 | 152.86 | 0 | 0 | all_rank_fail_fast |
| fsdp | 2 | constant | 0.0003 | 107.472 | 154.414 | 0 | 0 | all_rank_fail_fast |
| fsdp | 2 | constant | 0.0003 | 107.514 | 154.586 | 0 | 0 | all_rank_fail_fast |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
