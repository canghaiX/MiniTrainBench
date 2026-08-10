## 生成的 Benchmark 结果

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
| all_to_all | 2 | equal | 1024 | 0.069 | 0.119 | ok |
| all_to_all | 2 | uneven | 1024 | 0.158 | 0.052 | ok |
| all_to_all | 2 | equal | 1048576 | 0.069 | 120.828 | ok |
| all_to_all | 2 | uneven | 1048576 | 0.067 | 125.734 | ok |
| all_to_all | 2 | equal | 16777216 | 0.540 | 248.403 | ok |
| all_to_all | 2 | uneven | 16777216 | 0.619 | 216.662 | ok |
| all_to_all | 4 | equal | 1024 | 0.101 | 0.163 | ok |
| all_to_all | 4 | uneven | 1024 | 0.065 | 0.253 | ok |
| all_to_all | 4 | equal | 1048576 | 0.126 | 133.483 | ok |
| all_to_all | 4 | uneven | 1048576 | 0.302 | 55.486 | ok |
| all_to_all | 4 | equal | 16777216 | 1.208 | 222.278 | ok |
| all_to_all | 4 | uneven | 16777216 | 2.793 | 96.100 | ok |
| all_to_all | 8 | equal | 1024 | 0.091 | 0.360 | ok |
| all_to_all | 8 | uneven | 1024 | 0.070 | 0.470 | ok |
| all_to_all | 8 | equal | 1048576 | 0.214 | 156.887 | ok |
| all_to_all | 8 | uneven | 1048576 | 0.519 | 64.630 | ok |
| all_to_all | 8 | equal | 16777216 | 2.514 | 213.555 | ok |
| all_to_all | 8 | uneven | 16777216 | 6.975 | 76.966 | ok |

小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。all-to-all 对应 MoE expert parallel 的 token dispatch/combine，可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。
