## 生成的 Benchmark 结果

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 ± 0.00 | 12.10 ± 0.03 | 2.12 ± 0.01 | 35531.35 ± 72.04 | 14.41 ± 0.03 | 481.06 ± 0.29 | 100.00% | - | - | 3 |
| ddp | 2 | bf16 | 0.04 ± 0.00 | 13.68 ± 0.03 | 2.03 ± 0.01 | 64552.83 ± 62.79 | 15.86 ± 0.02 | 630.13 ± 44.55 | 90.84% | - | - | 3 |
| ddp | 4 | bf16 | 0.04 ± 0.00 | 14.78 ± 0.16 | 2.19 ± 0.03 | 119848.51 ± 1423.86 | 17.09 ± 0.20 | 660.42 ± 1.94 | 84.33% | - | - | 3 |
| ddp | 8 | bf16 | 0.04 ± 0.01 | 15.05 ± 0.18 | 2.17 ± 0.03 | 237503.13 ± 2688.13 | 17.25 ± 0.19 | 660.42 ± 1.94 | 83.55% | - | - | 3 |
| deepspeed_zero2 | 1 | bf16 | 0.04 ± 0.00 | 19.99 ± 0.47 | 12.58 ± 0.04 | 15568.16 ± 244.37 | 32.90 ± 0.51 | 2616.74 ± 253.04 | 100.00% | -443.96% | 18.49 | 3 |
| deepspeed_zero2 | 2 | bf16 | 0.04 ± 0.00 | 19.06 ± 0.06 | 8.74 ± 0.26 | 37255.19 ± 131.14 | 27.49 ± 0.10 | 2350.75 ± 144.62 | 119.65% | -273.06% | 11.62 | 3 |
| deepspeed_zero2 | 4 | bf16 | 0.04 ± 0.00 | 18.77 ± 0.19 | 5.08 ± 0.03 | 85597.27 ± 645.61 | 23.93 ± 0.18 | 2217.97 ± 90.38 | 137.46% | -235.84% | 6.84 | 3 |
| deepspeed_zero2 | 8 | bf16 | 0.04 ± 0.00 | 19.18 ± 0.15 | 3.56 ± 0.09 | 178333.32 ± 641.15 | 22.97 ± 0.08 | 2151.80 ± 64.65 | 143.19% | -225.83% | 5.72 | 3 |
| deepspeed_zero3 | 1 | bf16 | 0.05 ± 0.00 | 65.95 ± 1.81 | 12.13 ± 2.32 | 6546.87 ± 332.28 | 78.41 ± 4.11 | 2819.95 ± 1068.54 | 100.00% | -486.20% | 64.00 | 3 |
| deepspeed_zero3 | 2 | bf16 | 0.05 ± 0.00 | 67.90 ± 0.26 | 13.97 ± 0.17 | 12589.47 ± 90.53 | 81.34 ± 0.58 | 2418.26 ± 923.83 | 96.15% | -283.77% | 65.48 | 3 |
| deepspeed_zero3 | 4 | bf16 | 0.06 ± 0.00 | 70.83 ± 0.38 | 15.28 ± 0.29 | 24163.34 ± 143.57 | 84.76 ± 0.50 | 2243.80 ± 852.14 | 92.27% | -239.76% | 67.67 | 3 |
| deepspeed_zero3 | 8 | bf16 | 0.06 ± 0.00 | 73.02 ± 0.47 | 15.40 ± 0.45 | 46778.61 ± 251.62 | 87.56 ± 0.47 | 2157.75 ± 816.70 | 89.31% | -226.73% | 70.32 | 3 |

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

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
