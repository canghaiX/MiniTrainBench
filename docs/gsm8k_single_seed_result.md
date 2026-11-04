# Qwen3 HF reference 单 seed GRPO 完整验收

记录时间：2026-11-04T17:56:21.011353+08:00。全部系统验收通过。

固定 seed 2026；256 题 GSM8K 子集，每题 8 completions，最多 256 token；HF 增量 cache、BF16 autocast、双卡 DDP、16 步离线 GRPO 更新。前后评测使用同一 SGLang 0.5.18/FA3 与 greedy 配置，完整 test 1,319 题。

| 指标 | 实测结果 |
|---|---|
| 初始 test EM | 677/1,319 (51.3268%) |
| 更新 test EM | 677/1,319 (51.3268%) |
| EM 变化 | +0.0000 个百分点 |
| 全 token 最大 logprob 误差 | 0.0，门槛 0.5 |
| 正式更新 | 16 步，参数指纹变化，DDP rank 一致 |
| 最终 DCP 回读 | CPU 回读模型与 TrainState，模型摘要与正式训练报告一致 |
| 训练响应 token/s | 12.0917 |
| 峰值 allocated 显存 | 26.64 GiB（28,607,769,600 bytes） |

吞吐为串行 HF correctness reference 训练口径，不代表 serving 优化。只有一个 sampling seed，seed 标准差未定义，不能声称多 seed 稳定收益；子集训练和固定 token budget 不作为标准 benchmark 成绩。

本轮 EM 没有提升。参数指纹变化证明实际执行了更新；相同 EM 不能据此解释为特定原因，也不构成算法收益。正式训练从初始模型执行 16 步，没有在中途恢复；双卡 optimizer resume 单独由 smoke 验证，最终 step-16 DCP 只做模型与 TrainState 回读。

[轻量原始摘要](benchmark_results/gsm8k_hf_reference_single_seed_2026.json)保存模型、tokenizer、官方数据校验值、源码哈希、训练命令及软件版本，并包含更新/恢复 smoke 和清理记录。HF rollout 与训练在宿主环境运行；容器指纹只对应 SGLang 评测。训练与 rollout 当时存在未提交改动，原始 dirty 状态及 `provenance.complete=false` 如实保留，不声称由 clean commit 或完整容器构建复现。

smoke 中间权重与 DCP 已清理，估计释放约 19 GiB（未记录精确字节数）；完整验收后删除正式 DCP 15,553,720,750 bytes（约 14.49 GiB）。本轮目录剩余约 2.3 GiB，保留一份更新权重、rollout、已校验数据和小体积报告，不保留重复 checkpoint。评测容器已移除，GPU 已释放。公开仓库只提交轻量摘要和文档。

简历表述：构建 Qwen3-0.6B 离线 GRPO 系统验证链路，统一 HF 增量采样与训练概率口径，通过全 token logprob gate、双 A100 BF16/DDP 更新及 DCP 恢复 smoke；完成 seed 2026 的 16 步更新与 1,319 题前后评测，EM 实测变化 +0.0000 个百分点。

复现入口与历史运行快照见 [单 seed 运行记录](gsm8k_single_seed_run.md)。本轮完整回归：MiniTrainBench 84 passed、InferKernelLab 100 passed；改动 Python 的 Ruff/format、shell 语法与 JSON 校验通过。
