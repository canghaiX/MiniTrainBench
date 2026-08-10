## 生成的 Benchmark 结果

### 训练

| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | Tokens/sec | Step time (ms) | 最大显存 (MB) | 扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 0.04 ± 0.00 | 12.09 ± 0.01 | 2.10 ± 0.01 | 35598.71 ± 16.58 | 14.38 ± 0.01 | 481.06 ± 0.29 | 100.00% | - | - | 3 |
| ddp | 2 | bf16 | 0.04 ± 0.00 | 13.71 ± 0.02 | 2.04 ± 0.02 | 64386.95 ± 139.70 | 15.90 ± 0.03 | 660.42 ± 1.94 | 90.43% | - | - | 3 |
| ddp | 4 | bf16 | 0.04 ± 0.00 | 14.93 ± 0.39 | 2.14 ± 0.03 | 119600.88 ± 2876.79 | 17.13 ± 0.42 | 630.13 ± 44.55 | 83.99% | - | - | 3 |
| ddp | 8 | bf16 | 0.04 ± 0.00 | 15.70 ± 0.66 | 2.23 ± 0.05 | 227679.52 ± 8236.01 | 18.01 ± 0.64 | 659.42 ± 1.61 | 79.95% | - | - | 3 |
| fsdp | 1 | bf16 | 0.04 ± 0.00 | 30.05 ± 0.00 | 2.52 ± 0.03 | 15636.38 ± 17.74 | 32.74 ± 0.04 | 657.21 ± 144.96 | 100.00% | -36.62% | 18.36 | 3 |
| fsdp | 2 | bf16 | 0.04 ± 0.00 | 31.12 ± 0.32 | 1.33 ± 0.02 | 31728.64 ± 310.85 | 32.28 ± 0.31 | 274.90 ± 0.94 | 101.46% | 58.38% | 16.37 | 3 |
| fsdp | 4 | bf16 | 0.05 ± 0.00 | 31.49 ± 0.64 | 1.07 ± 0.03 | 63568.61 ± 1247.34 | 32.23 ± 0.64 | 210.51 ± 1.71 | 101.64% | 66.59% | 15.10 | 3 |
| fsdp | 8 | bf16 | 0.04 ± 0.00 | 31.69 ± 0.34 | 1.02 ± 0.01 | 127114.63 ± 1345.54 | 32.23 ± 0.34 | 172.83 ± 1.24 | 101.62% | 73.79% | 14.21 | 3 |

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

### 通信

| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| all_reduce | 8 | - | 1024 | 0.045 | 0.091 | ok |
| all_gather | 8 | - | 1024 | 0.201 | 0.163 | ok |
| reduce_scatter | 8 | - | 1024 | 0.059 | 0.558 | ok |
| all_to_all | 8 | equal | 1024 | 0.070 | 0.471 | ok |
| all_to_all | 8 | uneven | 1024 | 0.060 | 0.552 | ok |
| all_reduce | 8 | - | 1048576 | 0.117 | 35.734 | ok |
| all_gather | 8 | - | 1048576 | 0.334 | 100.452 | ok |
| reduce_scatter | 8 | - | 1048576 | 0.244 | 137.769 | ok |
| all_to_all | 8 | equal | 1048576 | 0.214 | 157.144 | ok |
| all_to_all | 8 | uneven | 1048576 | 0.492 | 68.171 | ok |
| all_reduce | 8 | - | 16777216 | 0.721 | 93.115 | ok |
| all_gather | 8 | - | 16777216 | 3.305 | 162.430 | ok |
| reduce_scatter | 8 | - | 16777216 | 2.214 | 242.540 | ok |
| all_to_all | 8 | equal | 16777216 | 2.428 | 221.119 | ok |
| all_to_all | 8 | uneven | 16777216 | 6.864 | 78.219 | ok |

小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。all-to-all 对应 MoE expert parallel 的 token dispatch/combine，可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。
