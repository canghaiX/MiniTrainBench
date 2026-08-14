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

### Failure Handling

| 故障类型 | 检测方式 | 自动恢复 | 恢复模式 | Checkpoint 未变 | 恢复 checkpoint | Global step | Tokens seen | 状态 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| checkpoint_resume_exact | checkpoint verify | 是 | - | - | /workspace/results/fault_tolerance/interrupted/step_00000003 | 3 | 24 | ok |
| config_mismatch | metadata fingerprint / config 校验 | 否 | - | - | - | - | - | rejected |
| nan_loss | all-rank loss/gradient finite reduction | 否 | - | - | step_00000001 | 1 | 8 | detected |
| rank_crash | launcher exit code / 进程组退出 | 否 | - | - | - | - | - | documented_not_injected |
| communication_timeout | NCCL watchdog / doctor connectivity | 否 | - | - | - | - | - | documented_not_injected |
| half_checkpoint | latest/READY 扫描 | 是 | - | - | step_00000003 | - | - | ok |

该表覆盖最小故障模型：精确 resume、半成品 checkpoint 跳过、配置不匹配拒绝、NaN、rank crash 和通信 timeout 的检测边界。
