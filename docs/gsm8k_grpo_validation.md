# Qwen3/SGLang GSM8K 实验验收（2026-11-03）

本页保留历史 SGLang 行为策略的失败预检与后续 HF smoke 记录。当前 seed 2026 的 HF reference 实验进度、已验证指标和复现命令见 [单 seed 验收记录](gsm8k_single_seed_run.md)；以下历史评测不能作为本轮初始/更新模型的完整评测结果。

**实现与组件 smoke 已通过，完整三 seed BF16 训练闭环尚未通过验收。**
全部 256 × 8 rollout 的 logprob 预检失败，正式实验在 optimizer 更新前停止，
因此没有三个更新模型的 test EM、均值/标准差或训练提升结论。
原始摘要见 [`gsm8k_grpo_validation.json`](gsm8k_grpo_validation.json)。

## 固定条件

- 本地 Qwen3-0.6B；模型和 tokenizer SHA-256 写入 JSON，未下载权重。
- GSM8K main train 按 seed 2026 固定选 256 题；完整官方 test 1,319 题。
  源文件 SHA-256 校验和 MIT 许可记录在数据 manifest。
- A100-SXM4-40GB，推理独立一张卡，训练使用两张卡。
- SGLang 0.5.18，FA3，`--rl-on-policy-target fsdp`，PyTorch sampler；
  容器 Torch 2.13.0/CUDA 13.0、Transformers 5.12.1。
- HF trainer 为 Torch 2.6.0/CUDA 12.4、Transformers 4.57.6；SDPA、BF16 autocast。
- temperature 0.8，top-p 1、禁用 top-k、每题 8 个回答、最多 256 token，保留 EOS。
  每个请求以 base seed 和序号的哈希派生 seed，避免相邻 seed 间复用随机流。
- 最大绝对 logprob 误差门槛固定为 0.5，未为通过实验而提高门槛。

## 完整 rollout 预检

| Rollout seed | Token 数 | Max error | P99 error | Mean error | 结果 |
|---|---:|---:|---:|---:|---|
| 2026 | 422,994 | 0.652631 | 0.155666 | 0.011141 | 失败，无参数更新 |
| 2027 | 419,699 | 1.254301 | 0.154289 | 0.011157 | 失败，无参数更新 |
| 2028 | 425,292 | 0.811586 | 0.153992 | 0.011123 | 失败，无参数更新 |

三个 rollout 的本地端到端收集吞吐约为 9,946、10,088、10,056 token/s。
这些数包含 HTTP 收集开销，不能替代统一负载的 serving benchmark。
约 0.38%–0.41% 的 token 初始 importance ratio 已在 [0.8, 1.2] 外，
故不能只用低平均误差来忽略尾部差异。最差 token 的题号、位置和双方 logprob
都记录在 JSON 中。

新增的固定 token 双端诊断确认：rollout 生成 logprob 与同一 SGLang 0.5.18 sampler 路径下的 teacher-forced 目标 token 分数，在所测位置逐项完全相同（64 个常规抽样位置及三个真实全量预检最差位置，误差均为 0）。因此误差不在 rollout logprob 序列的 token/位置记录或 sampler 输出复用。

偏差位于 SGLang 与 HF 模型前向分数之间。三个真实最差位置在 SGLang 容器 HF 5.12.1/Torch 2.13 上的误差分别为 0.196、0.546、0.724；在主机 HF 4.57.6/Torch 2.6 上为 0.196、1.087、1.767。说明 Transformers/Torch 版本会影响差异幅度，但升级/同容器运行不能全部解决。64 个抽样位置的 max error 为 0.087。当前证据指向实现/数值路径差异，尚未定位到具体层或算子；门槛仍未通过，禁止正式更新。

本轮在同一 SGLang FA3 server 上对三个 worst-case 各重测三次，SGLang 分数与 rollout 完全一致，重复分数 range 均为 0。新增 HF 前缀单 token 对照显示：seed 2027/2028 的全序列与前缀前向 logprob 自身相差 0.592/0.774；最终 hidden-state max 绝对差分别为 1.375/0.700/0.415。HF KV-cache 增量前向（FP32 权重 + BF16 autocast）仍有 0.762/0.569 两个样本超过门槛。该 SGLang HTTP 接口只返回最终 hidden state，不提供逐层激活；所以这组数据能定位到最终表示/概率路径差异，但不能声称已经找到具体层或算子。

HF FlashAttention 2/3 kernel 在固定容器中不可用，SGLang deterministic FSDP 模式也拒绝 `torch_native` attention backend。当前没有证据支持安全地修改 Qwen3 trainer 数值语义；因此本轮未改 trainer，也未启动小规模或正式 optimizer 更新。完整三 seed gate 仍失败。

进一步使用 SGLang 0.5.18 forward hooks 捕获 28 层的 norm、attention、MLP projections、MLP 和 block 输出，并与同一固定 token 的 HF 前缀前向对比。三个 worst-case 的 logprob 误差为 0.196、0.494、0.993；首次 block max error 超过 0.1 的层分别为 15、17、7。早期层输出接近，差异随深度累积，但首个分叉层不固定，不能归因于某个单独层/算子。逐层摘要和限制见 InferKernelLab 的 `docs/benchmark_results/qwen3_layerwise_logprob_diagnostic.json`。第三组仍超过 0.5，故不进行 trainer 改动或 optimizer 更新。

本轮进一步复现 trainer 的实际 group-batch 形状：每个 prompt 的 8 个 completion 右 padding 后一次前向，并对照同一 completion 的单条 HF 前向与 SGLang teacher-forced 分数。三个最差 token 的 HF group-batch 误差为 **0.653 / 1.254 / 0.812**；单条 HF 误差为 **0.196 / 1.087 / 1.767**；rollout 与 SGLang teacher-forced 在三处仍完全一致。group batching 在 seed 2026 放大误差 0.457，但 seed 2027、2028 的单条路径本身也超过 0.5，因此去掉 batching 不足以通过三个完整 rollout gate。本轮按一轮止损约束不改 trainer、不运行 optimizer。详细 token、logprob、rollout hash 与环境见 InferKernelLab 的 [`qwen3_trainer_batch_alignment_diagnostic.json`](../../InferKernelLab/docs/benchmark_results/qwen3_trainer_batch_alignment_diagnostic.json)；复现命令使用 `scripts/compare_grpo_logprobs.py --trainer-batch-diagnostic --preflight-json ...`。

后续实现了独立 `hf_reference` 路径：CPU tiny Qwen3 rollout 可复现并保留 EOS logprob；MiniTrainBench 增量 cache preflight 通过，actor 梯度可回传，Gloo DDP 参数一致，DCP 恢复与连续训练摘要一致。两个仓库完整 CPU pytest 已通过。A100 上 4 题 × 8 completion 的 smoke preflight 覆盖 1,024 个 token，最大误差为 0.0；但 32-token 上限令所有 completion 被截断、组内 reward 全相同，因此 optimizer 没有更新参数。该结果验证了数值路径，**没有通过实际更新/恢复 smoke**。按 fail-fast 原则，三组 256 × 8 full rollout gate 和正式更新尚未启动；`0.5` 门槛不变。

随后扩为 8 题 × 8 completion、最多 256 token：A100 串行生成 11,332 token 用时 337 秒（33.63 token/s）；双卡 BF16 preflight max/mean/P99 error 均为 0，最大误差门槛 0.5 通过。4 个 prompt group 有组内 reward 方差。一次 DDP 更新耗时 265.5 秒，参数实际变化、rank 一致，grad norm max 6.48，峰值分配显存 21,111,989,760 bytes（约 19.7 GiB）。从 step-1 DCP 恢复后到 step 2 通过同一 preflight，参数更新、rank 一致，峰值分配 25,658,156,032 bytes（约 23.9 GiB）。三组 256 × 8 full rollout gate 和正式训练仍未启动。

A100 更新 smoke 还发现增量路径的两个系统约束：不同 completion 长度使 DDP 每 token buffer broadcast 的 collective 序列不一致；把 8 条 completion 的 actor graph 同时保留则峰值达到 36.8 GiB 并 OOM。修复后对 incremental DDP 关闭逐 forward buffer broadcast，并逐 completion 反向累积等权梯度。双卡 smoke 已通过，观察峰值降至 19.7 GiB；CPU/Gloo 两 rank 测试验证了精确 DCP 恢复。

当前正在执行 seed 2026 的完整 HF reference 闭环：先复跑 smoke，然后生成 256 × 8 rollout，全量 max-error gate 通过后训练 16 步，并用同一 SGLang 0.5.18/FA3 配置评估初始和更新模型的 1,319 题 test。全部验收完成后，轻量摘要可导出到 MiniTrainBench 的 `docs/benchmark_results/gsm8k_hf_reference_single_seed_2026.json`；在此之前不发布成功摘要。单 seed 的 EM 方差不定义，不会报告成 0，也不承诺指标提升。

## 已通过的较小闭环组件

另用 **4 题 × 8 个回答、最多 256 token** 做双卡 BF16 smoke：

- 完整响应 token 预检通过，max error 为 0.409802。
- 执行 2 步 DDP 更新，参数实际改变；两个 rank 参数摘要相同。
- 从第 1 步 DCP checkpoint 恢复到第 2 步，与连续训练的最终参数 SHA-256 完全一致。
- 峰值分配显存 25,746,937,344 bytes（约 24 GiB）；短跑数字不作为稳定性能结论。
- 导出的模型重新通过 SGLang 加载，显式使用 BF16；全量 test 仍为 **677/1,319（51.33%）**，
  与初始模型相同，未观察到 EM 提升。初始和导出 tokenizer 对 1,319 个 prompt 的
  token IDs 逐题比较完全一致。

该 smoke 的训练集合与正式 256 题实验不同，且只使用单个 seed，不能用于替代正式验收。

## 完整测试集基线与边界

初始模型本地 greedy EM 为 **677/1,319 = 51.33%**；同一 no-thinking chat 提示、
最多 256 token，最后标记答案优先、无标记则取最后数字。它是本地完整测试集基线，
不是标准 benchmark 成绩，也不是训练后的收益。

前一轮完整 pytest 为 MiniTrainBench 57 passed、InferKernelLab 92 passed。本轮 InferKernelLab 完整 pytest 为 100 passed；MiniTrainBench 完整 pytest 使用 `PYTHONPATH=src` 运行并通过。
覆盖旧 Llama artifact、tiny Qwen3、GRPO loss、EOS mask、DDP、DCP resume、
GSM8K 数据校验/reward、SGLang HTTP 批量请求/响应、失败预检和错误汇总拦截。

复现入口与数据/概率约定见 [`qwen3_gsm8k_grpo_design.md`](qwen3_gsm8k_grpo_design.md)。
当前脚本按预期会在完整预检失败时停止。先解决尾部 logprob 差异，再执行三组
16 步更新及三个更新模型的全量重评；在此之前不发布更新后均值/标准差。
实验发生在功能提交前，JSON 保留当时父 commit、dirty 状态和源码摘要，
未将它伪装为 clean container provenance。
