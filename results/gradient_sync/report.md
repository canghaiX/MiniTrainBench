## 生成的 Benchmark 结果

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 2 | bf16 | 0.10 | 57.23 | 3.92 | 34339.65 | 59.64 | 656.15 | - | - | - | 1 |
| ddp | 2 | bf16 | 0.10 | 65.26 | 3.34 | 30118.36 | 68.00 | 655.06 | - | - | - | 1 |
| fsdp | 2 | bf16 | 0.10 | 129.40 | 2.75 | 15661.75 | 130.76 | 267.21 | - | 59.21% | 62.77 | 1 |
| fsdp | 2 | bf16 | 0.11 | 122.26 | 1.34 | 16540.01 | 123.82 | 267.21 | - | 59.21% | 55.82 | 1 |

扩展效率以同一策略的 1 卡吞吐为基准归一化。非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。

### Runtime 状态

| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | Latest | Keep last | Ready 数 | Resume path | Last checkpoint |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| ddp | 2 | DDPStrategy | 否 | 7 | 14336 | - | auto | last | 1 | - | - | 3 | 0 | - | - |
| ddp | 2 | DDPStrategy | 否 | 7 | 14336 | - | every | every | 4 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 7 | 14336 | - | auto | every | 4 | - | - | 3 | 0 | - | - |
| fsdp | 2 | FSDPStrategy | 否 | 7 | 14336 | - | last | last | 1 | - | - | 3 | 0 | - | - |

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
