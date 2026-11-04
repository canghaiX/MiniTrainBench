# 简历项目素材与证据边界

记录日期：2026-11-04。本页只整理已有实现和实测结果，不把正在运行的实验计入完成成果。适合训练 Infra、推理 Infra 与 RL Infra 实习的项目描述如下；实际投递时每个项目保留三到四条即可。

## MiniTrainBench：分布式训练与可靠性验证

技术栈：Python、PyTorch Distributed、DDP/FSDP、DeepSpeed、NCCL/Gloo、Distributed Checkpoint、PyTorch Profiler。

可直接用于简历：

- 构建可复现的分布式 GPT-like 训练 benchmark，支持 DDP/FSDP 与独立 DeepSpeed ZeRO-2/3 对照，覆盖 1/2/4/8 卡、BF16、梯度累积、激活检查点及吞吐/显存/通信分析，记录源码、环境和启动命令。
- 在单机 8×A100 BF16 显存压力实验中，731.1M 参数配置的 FSDP 峰值 allocated 显存由 DDP 的 16,904 MB 降至 1,844 MB，下降约 89.1%；2.60B 配置下 DDP OOM，FSDP 成功运行。该矩阵为合成数据短程压力测试，不能外推为长期预训练性能。
- 实现 DCP 检查点发布、READY/latest、保留策略及逐 rank RNG 状态恢复；通过连续训练与中断恢复的模型、优化器、scheduler、TrainState、RNG 对照，并补充真实 worker SIGKILL、非有限值与半成品 checkpoint 的故障验证。
- 实现多层 Transformer TP/DP 与 vocab-parallel causal LM，覆盖 Column/Row Parallel、分片 embedding/LM head、自定义 autograd cross entropy，以及 logits、局部梯度和更新后 DP 参数一致性校验。

证据：[显存压力矩阵](../results/memory_pressure/report.md)、[恢复验证](../results/fault_tolerance/verification.json)、[Transformer TP/DP](transformer_tp_dp_design.md)、[Vocab-parallel LM](vocab_parallel_lm_design.md)。

需要讲清楚：FSDP 省的是哪些状态的显存；梯度累积如何减少同步；TP replicated/sharded 参数如何归约；为什么 checkpoint 需要 RNG 与原子发布。ZeRO 是框架适配对照，不能表述成自行实现 ZeRO；故障后恢复有手工重启步骤，不能写成自动弹性容错。

## InferKernelLab：Paged KV Cache 与推理路径优化

技术栈：Python、PyTorch、Triton、CUDA、Transformers、SDPA、SGLang HTTP API。

可直接用于简历：

- 实现支持多层 decoder、GQA/MQA 和 ragged context 的 paged KV cache、block allocator 与 block table，开发直接读取分页缓存的 Triton decode attention，并以 PyTorch reference 和 HF eager oracle 验证 logits 与 greedy token 一致性。
- 定位并消除 decode runtime 重复构造 metadata 的开销，通过预分配及复用 block table/context metadata，在 A100 FP16、B=4/16、context=128/512 的既定矩阵上，使原 Triton backend 的 decode-step P50 降幅中位数约为 12.1%。
- 实现预算分块与批量 Prefill，合并 embedding、QKV、RoPE、MLP 和 KV 写入；在 A100 BF16 两层 tiny Llama、B2×128 下，Prefill P50 从 746.3 ms 降至 4.51 ms，greedy token match 为 100%（warmup 10、测量 50 次）。基线是本项目原逐 token Python 路径。

证据：[metadata 与 grouped 对照](https://github.com/canghaiX/InferKernelLab/blob/main/docs/benchmark_results/grouped_gqa_tensorcore_20260927.md)、[批量 Prefill A/B](https://github.com/canghaiX/InferKernelLab/blob/main/docs/benchmark_results/prefill_ab_afd68e9.md)。B4×512 的 Prefill P50 另由 6,013.4 ms 降至 5.51 ms；两个形状均通过 correctness。这里的明显改善主要来自消除逐 token Python/model dispatch，不能当作相对成熟 serving 引擎的加速比。

需要讲清楚：逻辑与物理 block 的映射、causal mask 与历史 chunk 的关系、online softmax 的数值稳定性，以及 attention microbenchmark、decode-step 和整模型 serving 吞吐的区别。grouped GQA kernel 尚无稳定端到端收益，12.1% 属于 metadata 复用，不能归因于 grouped kernel。未获得 Nsight counter 时，不声称实测 DRAM 带宽。

## 面向 RL Infra 的跨项目补充

当前可以写：

> 构建 Qwen3-0.6B 离线 GRPO 校验路径，统一 HF 增量 KV-cache 采样与训练打分口径；在 GSM8K 固定 256 题、每题 8 completion 的 419,959-token 全量预检中，最大 logprob 误差为 0（门槛 0.5）；完成双 A100 BF16/DDP 更新、rank 参数一致性及 DCP 恢复 smoke。

实现还包括温度/EOS 对齐、模型/tokenizer/数据和运行时指纹、失败 gate 阻止更新，以及 checkpoint 保留和中间产物清理。诊断覆盖 SGLang sampler、teacher-forced score、HF full-sequence/prefix/cache 和逐层激活对照。

证据：[单 seed 验收记录](gsm8k_single_seed_run.md)、[轻量验收快照](benchmark_results/gsm8k_hf_reference_progress_20261104.json)、[设计](qwen3_gsm8k_grpo_design.md)。正式 16 步训练及 1,319 题前后评测尚未全部验收，因此暂不写“已完成 GSM8K 训练评测闭环”或“EM 提升”。HF reference 是串行 correctness 基线；SGLang 当前作为独立 serving/评测对照，不声称其 rollout 已与 trainer 完成概率对齐。

## 投递时如何取舍

训练 Infra 优先展示显存压力、TP/DP 梯度与精确恢复；推理 Infra 优先展示 KV cache、metadata 复用及批量 Prefill；RL Infra 在这两项基础上补充概率预检与分布式更新恢复。简历正文突出实现、瓶颈、改动和可核验结果，弱化功能清单；详细测试条件和未完成项保留在仓库报告中。

本轮完整回归结果为 MiniTrainBench 84 passed、InferKernelLab 100 passed。单测数量是回归证据，不能代替 GPU 性能、数值正确性和真实实验验收。
