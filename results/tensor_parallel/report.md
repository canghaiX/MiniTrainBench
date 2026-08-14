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

### Tensor Parallel 正确性

| TP degree | Device | In | Out | Forward max error | Grad max error | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | cuda:0 | 1024 | 4096 | 3.8147e-05 | 0.000137329 | ok |

toy TP 校验把 ColumnParallelLinear 和 RowParallelLinear 与单卡 reference 对齐，用于说明 Megatron-style tensor parallel 的切分语义和梯度聚合路径。

### Megatron-style Toy Runtime 正确性

| 类型 | TP degree | Device | Shape | Forward max error | Grad max error | Collectives | 通信量估算 (bytes) | 状态 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| TP MLP | 2 | cuda:0 | 1024->4096->1024 | 0.0012207 | 0.00292969 | 2 | 131072 | ok |
| Sequence Parallel | 2 | cuda:0 | seq=8, hidden=1024 | 0 | 1.90735e-06 | 3 | 73728 | ok |

TP MLP 展示 ColumnParallel + RowParallel 如何组成一段可反传的 Megatron-style MLP；Sequence Parallel 展示 LayerNorm/Dropout 在 sequence shard 上的 correctness 边界。
