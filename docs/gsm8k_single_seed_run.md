# Qwen3 HF reference 单 seed GRPO 验收

本轮固定 seed 2026、256 题 GSM8K 训练子集、每题 8 条 completion、最多 256 token。HF reference 采样与 trainer 共用单条增量 KV-cache 概率口径；SGLang 0.5.18/FA3 用于训练前后完整 test 的独立评测。

## 已验证证据（2026-11-04，Asia/Shanghai 运行中快照）

| 检查 | 实测结果 |
|---|---|
| 离线数据准备 | 官方源文件 SHA-256 重新校验，准备 256/1,319 题 |
| A100 smoke | 8 题 × 8 completion，11,332 token；max/mean/P99 logprob error 均为 0 |
| BF16 双卡更新 smoke | 参数指纹变化，DDP rank 一致 |
| DCP resume smoke | 从 step 1 恢复到 step 2，实际更新且 rank 一致 |
| 正式 rollout | 256 组、2,048 completion、419,959 token；logprob 全部有限；rollout SHA-256 `e4716033b3b16ee45fc0ef905ee726f037fc1e861181c118610cf055e01f5d97` |
| 全量 logprob gate | 419,959 token；max/mean/P99 error 均为 0.0，门槛 0.5，通过 |
| 正式 rollout 采样 | 12,657.60 秒，33.18 token/s，峰值 allocated 3,696,577,536 bytes |
| Reward | 150/256 个组存在组内 reward 差异 |
| 正式训练进度 | 已完成 12/16 步，仍在运行；不等同于完整训练验收 |
| 完整回归 | MiniTrainBench 84 passed，InferKernelLab 100 passed；改动 Python Ruff/format、shell 语法和 JSON 校验通过 |

采样吞吐属于串行 HF correctness reference，不作为 serving 性能结论。token 数、有限性或 reward 方差也不能替代完整 policy gate。

**本记录更新时，全量 logprob gate 已通过，正式 16 步更新正在运行；本轮初始/更新模型的 1,319 题评测尚未开始。没有本轮 EM 差值或训练收益结论。** 历史 SGLang rollout 的失败预检与历史小样本评测保留在 [历史验收记录](gsm8k_grpo_validation.md)，不能当作本轮结果。

本地收尾进程等待当前训练进程退出，再验证训练报告并接续完整评测，最后生成 `docs/gsm8k_single_seed_result.md`。成功时该文件记录完整 test EM、实际百分点变化和简历口径，并导出轻量 JSON；失败时只记录停止阶段，不生成闭环成功摘要。本页是运行中的证据快照，最终状态以该收尾报告与原始阶段 JSON 为准。收尾日志位于本轮结果目录的 `finalize.log`。

## 复现与结果导出

在 MiniTrainBench 根目录执行。OUT_ROOT 必须是不存在的新目录，数据脚本仅从已有本地源文件复制并校验，不联网补齐：

~~~bash
OUT_ROOT=results/gsm8k_hf_reference_seed2026_repro DATA_SOURCE_DIR=results/gsm8k_grpo/data INFER_ROOT=/data/InferKernelLab TRAIN_GPUS=0,1 HF_REFERENCE_GPU=0 SERVER_GPU=7 SEEDS=2026 bash scripts/run_gsm8k_hf_reference_cycle.sh
~~~

本轮实际产物位于本地 results/gsm8k_hf_reference_single_seed_2026。原始数据、rollout、导出权重和 checkpoint 不纳入 Git。顺序为 smoke → 正式 rollout → 全 token gate → 16 步更新 → 初始/更新模型评测 → summary。gate 要求误差全部有限且 max_abs_error ≤ 0.5；失败即停止。正式 DCP 每步保存，保留最新两个 checkpoint。

磁盘策略：smoke 更新和恢复均通过后删除其两份导出权重与 DCP，保留小体积报告和 rollout；正式训练期间只保留最近两个 checkpoint；完整评测和 summary 验收后删除正式 DCP，仅保留一份更新模型和复现所需的小文件。清理工具只处理本轮指定目录，拒绝符号链接，记录 `cleanup_smoke.json` / `cleanup_complete.json`。本轮已清理约 19 GB smoke 中间产物。容器日志限制为 10 MB，退出时移除本轮容器。

若训练已完成但后续流程中断，可以只接续评测，不重新生成 rollout 或更新参数：

~~~bash
OUT_ROOT=results/gsm8k_hf_reference_single_seed_2026 INFER_ROOT=/data/InferKernelLab SEEDS=2026 bash scripts/run_gsm8k_hf_reference_cycle.sh --evaluate-only
~~~

该入口会先校验 16 步训练、参数指纹变化、rank 一致性和 logprob gate。训练报告缺失或不合格时，启动评测前即拒绝继续。

完整流水线成功后，可导出轻量摘要：

~~~bash
PYTHONPATH=src python3 scripts/summarize_gsm8k_grpo.py --root results/gsm8k_hf_reference_single_seed_2026 --expected-seeds 1 --output results/gsm8k_hf_reference_single_seed_2026/summary.json --compact-output docs/benchmark_results/gsm8k_hf_reference_single_seed_2026.json
~~~

摘要要求 16 步实际更新、模型指纹变化、rank 一致、评测模型匹配和完整 test 数量一致。单 seed 标准差写为 null；EM 下降照实保留。缺少训练/评测结果或 gate 未通过时，拒绝生成成功摘要。实验运行时工作区包含已有未提交改动，provenance 保留 dirty 状态与源码 SHA-256，不声称 clean commit 可复现。

## 当前简历口径

统一 Qwen3 HF 增量采样/训练 logprob 口径，在双 A100 上完成 BF16 GRPO 更新与 DCP 恢复 smoke，参数实际变化且 rank 一致；seed 2026 的 256 × 8 正式 rollout 共 419,959 token，全量 gate 最大误差为 0。完整单 seed 训练评测仍待验收，暂不写成完整闭环或效果提升。轻量原始字段见 [验收快照](benchmark_results/gsm8k_hf_reference_progress_20261104.json)。
