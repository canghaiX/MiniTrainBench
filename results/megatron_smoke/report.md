# Megatron-LM 8 卡 Smoke / TP-PP-DP 矩阵

Megatron 源码由用户通过 `MEGATRON_DIR` 提供；本目录只保存命令、版本、日志解析结果和失败原因。

**警告：本报告包含兼容性 smoke，性能指标不可横向比较。**
原因：compatibility_smoke, concurrent_gpu_compute_processes, non_ngc_fallback_environment

| 配置 | 环境 | TP | PP | DP | Repeat | 状态 | 性能可比 | Tokens/sec | Step (ms) | 设备峰值显存 (MB) | 理论 bubble proxy | 失败原因 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| tp1_pp1 | pytorch_2_10_official_fallback | 1 | 1 | 8 | 1 | success | no | - | - | - | 0.0 | - |
| tp2_pp1 | pytorch_2_10_official_fallback | 2 | 1 | 4 | 1 | success | no | - | - | - | 0.0 | - |
| tp4_pp1 | pytorch_2_10_official_fallback | 4 | 1 | 2 | 1 | success | no | - | - | - | 0.0 | - |
| tp2_pp2 | pytorch_2_10_official_fallback | 2 | 2 | 2 | 1 | success | no | - | - | - | 0.2 | - |
| tp1_pp4 | pytorch_2_10_official_fallback | 1 | 4 | 2 | 1 | success | no | - | - | - | 0.42857142857142855 | - |

`Tokens/sec` 由 global batch、sequence length 与测量 step 均值推导；设备显存来自
独占 GPU 条件下的 `nvidia-smi` 设备级采样，不等同于 PyTorch allocator 指标。
Pipeline bubble 仅给出 fill-drain 理论 proxy；没有 trace 时不声称观察到 idle。
兼容性 smoke 的原始指标保留在 JSON 供审计，但 Markdown 不展示为性能结论。
