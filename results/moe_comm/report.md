## 生成的 Benchmark 结果

### 实验环境

| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | Image ID | Base image | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.10.0+cu130 | 13.0 | 91501 | 2.28.9 | 580.173.02 | NVIDIA A100-SXM4-40GB | 048ca3ed54df | sha256:95dcdfd0b564 | pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b | 完整 |

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| all_to_all | 2 | equal | 1024 | 0.060 | 0.136 | ok |
| all_to_all | 2 | uneven | 1024 | 0.060 | 0.136 | ok |
| all_to_all | 2 | equal | 1048576 | 0.071 | 118.179 | ok |
| all_to_all | 2 | uneven | 1048576 | 0.066 | 127.341 | ok |
| all_to_all | 2 | equal | 16777216 | 0.533 | 251.757 | ok |
| all_to_all | 2 | uneven | 16777216 | 0.618 | 217.027 | ok |
| all_to_all | 4 | equal | 1024 | 0.073 | 0.224 | ok |
| all_to_all | 4 | uneven | 1024 | 0.062 | 0.263 | ok |
| all_to_all | 4 | equal | 1048576 | 0.126 | 133.606 | ok |
| all_to_all | 4 | uneven | 1048576 | 0.288 | 58.283 | ok |
| all_to_all | 4 | equal | 16777216 | 1.207 | 222.342 | ok |
| all_to_all | 4 | uneven | 16777216 | 2.947 | 91.082 | ok |
| all_to_all | 8 | equal | 1024 | 0.080 | 0.409 | ok |
| all_to_all | 8 | uneven | 1024 | 0.064 | 0.516 | ok |
| all_to_all | 8 | equal | 1048576 | 0.213 | 157.465 | ok |
| all_to_all | 8 | uneven | 1048576 | 0.493 | 68.097 | ok |
| all_to_all | 8 | equal | 16777216 | 2.485 | 216.052 | ok |
| all_to_all | 8 | uneven | 16777216 | 6.913 | 77.666 | ok |

小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。all-to-all 对应 MoE expert parallel 的 token dispatch/combine，可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。
