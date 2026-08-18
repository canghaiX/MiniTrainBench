# MiniTrainBench 深度面试问答

这份文档用于面试前复盘，包含基础热身题、项目深挖题和投递场景面经。深挖题不追求数量堆满，而是模拟真实面试里“围绕一个点继续追问”的节奏。

回答时建议遵守三条原则：

- 先讲项目目标，再讲实现细节。
- 先讲证据，再讲结论。
- 主动说明边界，不把 compatibility smoke 说成正式 benchmark。

## 0. 基础热身题

这一节更像面经里的开场题，重点是把基础概念说清楚，不要一上来就陷入实现细节。

### B1. 面试官：什么是反向传播？

答：反向传播就是根据 loss 对模型参数求梯度的过程。前向传播先算出预测和 loss，反向传播再沿着计算图把梯度传回每一层，最后 optimizer 根据梯度更新参数。

在 PyTorch 里，一般是 `loss.backward()` 触发 autograd 自动完成反向传播。

### B2. 面试官：什么是 optimizer？

答：optimizer 是参数更新器。它读取每个参数的梯度，然后按照 AdamW、SGD 这类算法更新参数。

在本项目里默认使用 AdamW，所以 checkpoint 不能只保存模型参数，还要保存 optimizer state。因为 AdamW 里有一阶矩、二阶矩等状态，它们会影响后续训练轨迹。

### B3. 面试官：optimizer 和 scheduler 有什么区别？

答：optimizer 决定“怎么用梯度更新参数”，scheduler 决定“每一步用多大学习率”。optimizer 关心参数更新规则，scheduler 关心学习率随 step 如何变化。

所以精确 resume 时，两者都要恢复。只恢复 optimizer，不恢复 scheduler，后续学习率可能会错。

### B4. 面试官：什么是 learning rate？

答：learning rate 是每次参数更新的步长。太大可能训练不稳定，太小可能收敛很慢。

在训练系统里，learning rate 通常不是固定写死的，而是由 scheduler 按 step 推进。本项目支持 constant 和 cosine 两种 scheduler。

### B5. 面试官：什么是 batch size？

答：batch size 是一次训练 step 里每个 rank 或整个训练任务处理多少样本。分布式训练里要区分 local batch size 和 global batch size。

本项目里的 global batch size 等于 `batch_size_per_rank * world_size * grad_accum_steps`。

### B6. 面试官：step、iteration、epoch 有什么区别？

答：step 或 iteration 通常指一次 optimizer update。epoch 指完整遍历一遍数据集。

这个项目使用 synthetic token，不依赖真实数据集，所以更关注 global step，而不是 epoch。因为 benchmark 和 checkpoint 都围绕 step 组织。

### B7. 面试官：什么是 gradient accumulation？

答：gradient accumulation 是先跑多个 micro-batch，把梯度累积起来，再做一次 optimizer step。它常用于显存不够放大 batch 的情况。

关键点是每个 micro-batch 的 loss 要除以 accumulation steps，否则梯度会被放大。

### B8. 面试官：什么是 micro-batch？

答：micro-batch 是 gradient accumulation 里的一个小批次。多个 micro-batch 累积后，形成一次等效的大 batch 更新。

在分布式训练里，micro-batch 会影响显存、通信频率和 pipeline 调度。

### B9. 面试官：什么是 dropout？

答：dropout 是训练时随机丢弃部分激活，减少过拟合的一种方法。它只在训练阶段启用，推理阶段通常关闭。

dropout 依赖随机数，所以本项目做精确 resume 时必须保存每个 rank 的 RNG state。

### B10. 面试官：什么是 LayerNorm？

答：LayerNorm 是对每个样本内部的 hidden 维度做归一化。Transformer 里常用 LayerNorm，因为它不依赖 batch 统计，更适合变长序列和分布式训练。

和 BatchNorm 不同，LayerNorm 不需要跨 batch 维护 running mean 和 running variance。

### B11. 面试官：FP32、FP16、BF16 有什么区别？

答：FP32 精度高但显存和计算开销更大。FP16 更省显存、速度快，但数值范围小，训练时经常需要 GradScaler。BF16 数值范围接近 FP32，显存接近 FP16，所以大模型训练里很常见。

本项目主线使用 BF16/FP32，没有实现 FP16 GradScaler。如果未来支持 FP16，scaler state 也应该进入 checkpoint。

### B12. 面试官：什么是梯度裁剪？

答：梯度裁剪是当梯度范数超过阈值时，把梯度缩放到阈值以内。它常用于防止梯度爆炸。

本项目会记录 grad norm 和 clipped steps，这样可以知道训练是否稳定，以及裁剪是否真的发生。

### B13. 面试官：什么是 checkpoint？

答：checkpoint 是训练状态快照，用来在中断后恢复训练。最基础的 checkpoint 可能只保存模型参数，但严肃训练里还要保存 optimizer、scheduler、step、RNG 等状态。

本项目强调的是精确 checkpoint，不只是能继续跑，还要能验证恢复后和连续训练一致。

### B14. 面试官：什么是 resume？

答：resume 是从 checkpoint 恢复训练。它不只是 load model，还要恢复训练进度、优化器状态、学习率状态和随机状态。

如果这些状态不完整，训练可能能继续，但不能保证和没中断时一样。

### B15. 面试官：为什么训练要固定 seed？

答：固定 seed 是为了让随机初始化、数据生成和随机算子尽量可复现。没有固定 seed，重复实验的差异很难解释。

本项目里 synthetic data 和模型初始化都依赖 seed，这也是 repeat 和 checkpoint verify 能成立的基础。

### B16. 面试官：什么是 distributed training？

答：distributed training 是用多个进程、多张 GPU 或多台机器一起训练模型。常见目标是提升吞吐、放下更大模型，或者处理更大的 batch。

这个项目主要做单节点多 GPU 分布式训练，重点比较 DDP、FSDP 和 ZeRO。

### B17. 面试官：什么是 rank 和 world size？

答：rank 是分布式训练中每个进程的编号，world size 是总进程数。比如 8 卡单机通常有 8 个 rank，world size 是 8。

很多状态都要按 rank 区分，比如每个 rank 的 RNG state、local shard 和通信计时。

### B18. 面试官：什么是 DDP？

答：DDP 是 PyTorch 的 DistributedDataParallel。每个 rank 持有完整模型，处理不同数据，反向传播时通过 all-reduce 同步梯度。

它的优点是简单、吞吐通常好；缺点是每张卡都要放完整参数、梯度和 optimizer state。

### B19. 面试官：什么是 FSDP？

答：FSDP 是 Fully Sharded Data Parallel。它会把参数、梯度和 optimizer state 分片到不同 rank，降低单卡显存。

代价是 forward/backward 中会引入 all-gather、reduce-scatter 等通信，所以小模型上不一定比 DDP 快。

### B20. 面试官：什么是 all-reduce？

答：all-reduce 是一种 collective 通信。每个 rank 贡献一个 tensor，通信后每个 rank 都得到聚合结果。

DDP 梯度同步主要依赖 all-reduce。

### B21. 面试官：什么是 all-gather 和 reduce-scatter？

答：all-gather 是每个 rank 拿到所有 rank 的数据。reduce-scatter 是先聚合所有 rank 的数据，再把结果切分回每个 rank。

FSDP 经常用 all-gather 获取完整参数，用 reduce-scatter 聚合并分片梯度。

### B22. 面试官：什么是 all-to-all？

答：all-to-all 是每个 rank 都给其他 rank 发送不同的数据，同时也从其他 rank 接收数据。

MoE expert parallel 里，token 会按 expert 重新分发，所以 all-to-all 是核心通信模式。

### B23. 面试官：吞吐和延迟有什么区别？

答：吞吐关注单位时间处理多少 token 或样本，延迟关注一次操作花多长时间。训练 benchmark 常看 tokens/sec，通信 benchmark 常同时看 latency 和 bandwidth。

小 tensor 更容易受延迟影响，大 tensor 更容易体现带宽上限。

### B24. 面试官：什么是 profiler？

答：profiler 是性能分析工具，用来记录训练过程中时间花在哪里。它能帮助判断瓶颈在计算、通信、optimizer、CPU 调度还是数据准备。

本项目把 profiler 和正式 benchmark 分开，避免 profiling 开销污染吞吐结果。

### B25. 面试官：什么是 straggler？

答：straggler 是分布式训练中比其他 rank 慢的进程。同步训练必须等最慢 rank，所以一个 straggler 就可能拖慢整个 step。

本项目 profiler 报告里会看 rank min、p50、max 和 straggler ratio。

### B26. 面试官：什么是 OOM？

答：OOM 是 out of memory，表示显存或内存不够。大模型训练里 OOM 很常见，通常需要减 batch、开 activation checkpointing、使用 FSDP/ZeRO，或者调整模型规模。

本项目的 memory pressure 矩阵就专门记录了哪些配置成功、哪些配置 OOM。

### B27. 面试官：什么是 overfitting？

答：overfitting 是模型在训练集上表现很好，但泛化到新数据上表现差。常见缓解方式包括更多数据、正则化、dropout、early stopping 等。

不过本项目不以模型效果为主，不评价泛化能力，主要关注训练系统行为。

### B28. 面试官：为什么要做 warmup？

答：warmup 可以排除刚启动时的冷态影响，比如 CUDA context 初始化、kernel 缓存、内存分配和通信初始化。

本项目 benchmark 会把 warmup steps 和 measured steps 分开，正式统计只看 measured steps。

## 一、项目定位和整体设计

### Q1. 面试官：你先用一两分钟介绍一下这个项目。

答：MiniTrainBench 是一个面向训练 Infra 的最小分布式训练 runtime 和可复现实验套件。它不是单纯跑一个 GPT-like 模型，而是围绕 DDP、FSDP、DeepSpeed ZeRO、checkpoint/resume、Profiler、collective 通信、MoE all-to-all、toy Tensor Parallel 和故障恢复建立一条证据链。

项目使用 deterministic synthetic token，避免数据下载和预处理噪声，把重点放在训练系统本身的状态、同步、性能和恢复语义上。我会把它定位成“能跑、能恢复、能解释、能复查”的最小训练系统。

### Q2. 面试官：为什么你说它是 runtime，而不是一个 benchmark 脚本？

答：普通 benchmark 脚本通常只负责初始化模型、跑几步、输出吞吐。这个项目里有 `Trainer`、`TrainingConfig`、`TrainState`、`TrainingStrategy`、`CheckpointManager` 这些运行时组件，负责管理训练生命周期。

它不仅做 forward/backward，还管理 step 推进、gradient sync、LR scheduler、checkpoint 发布、resume 校验、repeat trial 和故障处理。所以我说它是 runtime，是因为它关心“训练如何长期、可控、可恢复地运行”，而不是只关心“这次命令能不能跑完”。

### Q3. 面试官：为什么你不直接做一个大而全的训练框架？

答：这个项目的目标不是复刻 Megatron、DeepSpeed 或生产训练平台，而是把训练系统里最核心、最容易出错的契约先做实。比如梯度同步什么时候发生，checkpoint 什么状态才算可恢复，resume 后 scheduler 和 RNG 是否一致，性能数字是否可复查。

如果一开始就做大而全，很容易把数据管线、多机调度、作业平台、推理、RLHF 都堆进去，但每个点都讲不深。最小 runtime 的好处是范围清楚，实验可控，每一个能力都有源码和结果证据支撑。

### Q4. 面试官追问：那这个“最小”会不会显得项目不够复杂？

答：不会，因为“最小”不是功能少，而是边界克制。项目覆盖了 DDP/FSDP/ZeRO、分布式 checkpoint、rank crash、Profiler、MoE all-to-all、toy TP/SP、memory pressure 和 provenance，这些都是训练 Infra 的核心问题。

复杂度不在页面或 API 数量，而在状态语义是否正确。比如 checkpoint 里少存一个 RNG state，训练能继续跑，但已经不能保证和连续训练一致；FSDP checkpoint 不能按普通 tensor 比较，否则分片状态会出错。

### Q5. 面试官：为什么使用 synthetic data，而不使用真实数据集？

答：因为这个项目主要验证训练 runtime，而不是验证数据集效果。synthetic token 可以完全由 seed、step、rank 决定，不依赖下载、不依赖 tokenizer、不依赖数据预处理，也不会受到 shuffle 或 IO 抖动影响。

这对 checkpoint/resume 特别重要。恢复后如果数据来源不确定，就很难判断状态差异来自 runtime 还是数据管线。使用 deterministic synthetic data 后，我能更清楚地说：如果连续训练和中断恢复不一致，问题大概率在模型、optimizer、scheduler、RNG 或 checkpoint 生命周期里，而不是数据集噪声。

### Q6. 面试官追问：真实训练里数据管线也很重要，你这里不用真实数据，会不会结论不完整？

答：结论确实有边界。这个项目不能声称覆盖真实数据吞吐、tokenizer 性能、远程存储 IO 或 dataloader worker 调优。

但这是有意为之。我的主线是 pretraining runtime 的状态和通信语义，不是端到端数据平台。synthetic data 把数据变量固定住，让 DDP/FSDP/ZeRO、checkpoint/resume 和 profiler 的结论更干净。真实数据管线可以作为后续扩展，但不应该污染当前 runtime contract 的验证。

### Q7. 面试官：这个项目最想展示你的什么能力？

答：主要展示三类能力。第一是分布式训练理解，比如 DDP/FSDP/ZeRO 的显存和通信差异，gradient accumulation 的同步语义，all-reduce/all-gather/reduce-scatter/all-to-all 的用途。

第二是训练 runtime 工程能力，比如状态管理、checkpoint 原子发布、精确 resume、故障边界和 CI。第三是性能工程能力，比如 repeat 统计、rank 诊断、Profiler trace、memory pressure 和 provenance。我不会把它讲成“我实现了一个大模型框架”，而是讲成“我把训练系统最关键的几个 contract 做成了可验证证据”。

### Q8. 面试官：你觉得这个项目最大的亮点是什么？

答：最大的亮点是它不只报 tokens/sec，而是把正确性、性能和故障恢复放在同一条证据链里。比如同样是 checkpoint，项目不是只保存模型权重，而是保存 model、optimizer、scheduler、TrainState 和每 rank RNG，并用 `checkpoint verify` 证明恢复后状态一致。

另一个亮点是边界诚实。Megatron 只做 compatibility smoke，就明确不展示性能数字；rank crash 是 manual restart，就不说成 TorchElastic 自动恢复；没有稳定多机资源，就不伪造多机 benchmark。

## 二、Runtime 架构和状态设计

### Q9. 面试官：你这个 runtime 的核心模块怎么拆？

答：核心是 `Trainer`、`TrainingConfig`、`TrainState`、`StepMetrics` 和 `TrainingStrategy`。`Trainer` 负责训练循环和生命周期；`TrainingConfig` 固化运行配置；`TrainState` 记录 global step、micro step、tokens seen、seed 和恢复来源；`StepMetrics` 记录每一步的耗时、loss、lr、grad norm；`TrainingStrategy` 隔离 DDP/FSDP 的包装和同步差异。

此外还有 `CheckpointManager` 管理保存、恢复、READY、latest 和 retention；`verification` 负责加载两份 checkpoint 做状态 digest；`profiler` 负责每 rank trace 和摘要；`communication` 负责 collective microbenchmark。

### Q10. 面试官追问：为什么要有 `TrainingStrategy`？直接 if strategy == ddp 不行吗？

答：可以写 if，但后期会让训练主循环越来越混乱。DDP 和 FSDP 的差别不只是包模型的方式不同，还包括是否需要 process group、默认 gradient sync 策略、`no_sync()` 的风险、grad norm clipping 的实现方式、state dict 语义。

我把这些差异放到 `TrainingStrategy` 里，让 `Trainer` 只关心“什么时候需要同步、什么时候做 step、什么时候保存状态”。这样以后加策略时，不需要在主循环里到处插分支。

### Q11. 面试官：`TrainingConfig` 为什么还要做 fingerprint？

答：checkpoint 恢复最怕“看起来能恢复，实际语义变了”。比如模型层数、batch size、gradient accumulation、precision、scheduler 或 gradient sync mode 变了，恢复后可能还能跑，但已经不是同一个训练过程。

`config_fingerprint` 的作用就是把关键训练语义压成一个可比较的摘要。恢复时如果 fingerprint 不匹配，就拒绝恢复。这样做比只检查路径或文件存在更安全。

### Q12. 面试官追问：为什么不是所有字段都进 fingerprint？

答：fingerprint 应该覆盖会影响训练语义的字段，比如策略、精度、模型配置、batch、accumulation、learning rate、scheduler、grad clipping、sync mode 和 seed。像输出路径、保存间隔、报告路径这类运行外壳，不应该影响 checkpoint 是否可恢复。

这个选择体现的是 contract 边界：哪些字段改变后会让后续参数轨迹变掉，哪些只是实验管理字段。过少会误恢复，过多会让本来兼容的 checkpoint 被拒绝。

### Q13. 面试官：`TrainState` 里为什么要同时有 `global_step`、`micro_step` 和 `tokens_seen`？

答：`global_step` 表示 optimizer step 的进度，是 scheduler 和 checkpoint 的主要时间轴。`micro_step` 表示 gradient accumulation 内部位置，便于描述训练状态和异常发生点。`tokens_seen` 是全局 token 进度，方便验证恢复前后是否跳过或重复消费 batch。

在这个项目里 checkpoint 通常在完整 step 后保存，所以恢复主要依赖 `global_step`。但把这些状态显式保存下来，可以让 runtime 状态更可审计。

### Q14. 面试官：为什么要把 learning rate 和 grad norm 都放进指标？

答：因为它们是训练稳定性的基本信号。learning rate 可以验证 scheduler 是否按 optimizer step 推进，尤其 resume 后要确认 `completed_steps` 和 `global_step` 一致。grad norm 可以观察梯度是否异常、是否发生裁剪、是否出现非有限值。

只看 loss 和 tokens/sec 不够。一个训练系统可能吞吐正常，但 scheduler 错了一步，或者某个 rank 出现 NaN 后还继续 optimizer step，这些都需要 runtime 指标暴露出来。

### Q15. 面试官：这个项目里 repeat 是怎么做的？

答：`repeat > 1` 时，每个 trial 会重新初始化模型、optimizer、TrainState 和 synthetic iterator，然后独立跑 warmup 和 measured steps。最后报告 mean、std、min、max。

我没有把 repeat 做成同一个训练过程里的连续窗口，因为那样不同窗口会混入 loss 曲线变化、optimizer state、缓存状态和 scheduler 变化。独立 trial 更接近 benchmark 统计意义上的重复实验。

### Q16. 面试官追问：repeat 和 resume 为什么互斥？

答：repeat 的语义是“独立性能试验”，resume 的语义是“继续某个训练状态”。如果两者混在一起，就很难解释每个 trial 的状态来源。

比如从 checkpoint 恢复后再 repeat 三次，是重新从同一个 checkpoint 开三次，还是接着训练三段？这会让 runtime 状态和性能统计都变得模糊。所以我把它们拆开：正式性能用 independent repeat，恢复正确性用 continuous vs interrupted verify。

## 三、DDP、FSDP 和梯度同步

### Q17. 面试官：DDP 和 FSDP 的核心区别是什么？

答：DDP 每个 rank 都保存完整模型参数、梯度和 optimizer state，backward 时通过 all-reduce 同步梯度。它实现简单，中小模型上吞吐通常更好，但显存占用随模型完整复制。

FSDP 会把参数、梯度和 optimizer state 分片，forward/backward 过程中通过 all-gather 和 reduce-scatter 临时组织计算。它能显著降低显存，但通信路径更复杂，小模型上不一定更快。这个项目的实验也体现了这个结论：小模型 DDP 吞吐更好，大模型压力下 FSDP 能把 DDP OOM 的场景跑通。

### Q18. 面试官追问：为什么 FSDP 小模型上可能更慢？

答：小模型的计算量不够大，FSDP 引入的参数 all-gather、reduce-scatter、hook 调度和分片管理开销会比较明显。DDP 只在 backward 同步梯度，通信模式更简单，所以小模型经常吞吐更高。

FSDP 的价值不是“永远更快”，而是“显存换可训练性”。当模型接近显存边界时，能不能跑起来比小模型吞吐更关键。

### Q19. 面试官：gradient accumulation 在 DDP 里有什么坑？

答：最常见的坑是虽然把 loss 除以 accumulation steps，但每个 micro-batch 的 backward 仍然触发 DDP all-reduce。这样数学上累积了梯度，但通信没有减少，性能收益会被吃掉。

正确做法是前面的 micro-batch 用 `model.no_sync()` 包住，只在最后一个 micro-batch 同步梯度。并且 `no_sync()` 要覆盖 forward 和 backward，因为 DDP reducer 的状态和 forward 也有关。

### Q20. 面试官追问：为什么只包 backward 不够？

答：DDP 在 forward 阶段会设置 reducer 的一些状态，决定后续 backward 时哪些 bucket 和参数需要同步。如果只在 backward 外面包 `no_sync()`，可能已经错过了 DDP 期望的上下文。

所以项目里是把完整 forward/backward 都放进 strategy 的 `gradient_sync_context` 中。这样语义更符合 PyTorch DDP 的使用方式。

### Q21. 面试官：那 FSDP 为什么不默认也这么做？

答：FSDP 的 `no_sync()` 和 DDP 不完全一样。FSDP 在 no_sync window 里可能保留未分片梯度，这会降低通信频率，但可能显著增加峰值显存。

所以项目里 DDP 的 `auto` 解析成 `last`，FSDP 的 `auto` 解析成 `every`。这是一个保守选择：DDP 默认追求减少重复 all-reduce，FSDP 默认优先显存安全。如果用户明确想用 FSDP `last`，也可以显式选择。

### Q22. 面试官：你怎么保证 gradient accumulation 的梯度尺度是对的？

答：每个 micro-batch 的 loss 会除以 `grad_accum_steps` 再 backward。这样累积 N 次后的梯度等价于一个更大的 batch 的平均梯度，而不是把梯度放大 N 倍。

如果不除以 N，learning rate 等效会变大，训练稳定性和不同设置之间的可比性都会受影响。

### Q23. 面试官：你怎么统计分布式 step time？

答：每个 rank 先本地测量 step time，然后跨 rank 取 max。同步训练下一步必须等最慢 rank，所以 rank 0 的本地时间不代表全局 step time。

CUDA 计时前后也需要 synchronize，因为 GPU kernel 是异步的。如果不同步，计时会低估实际耗时。

### Q24. 面试官追问：loss 为什么不也取 max？

答：loss 是训练数值指标，通常更适合取 mean 来表示全局平均情况。step time 和显存是资源瓶颈指标，应该取 max，因为最慢 rank 和最高显存 rank 决定训练是否可持续。

这体现了不同指标的聚合语义不同：性能瓶颈看最坏，数值状态看聚合。

### Q25. 面试官：你怎么解释 DDP 8 卡不是 8 倍加速？

答：同步训练里会有通信、kernel launch、optimizer、框架调度、负载不均和串行部分。尤其小模型计算量有限，通信和固定开销占比更高，所以不会线性扩展。

我不会只说“网络慢”。更准确的说法是，扩展效率受计算通信比例、collective 调用频率、消息大小、rank straggler 和 overlap 共同影响。项目用 repeat 结果和 profiler 辅助解释这些因素。

### Q26. 面试官：这个项目里 BF16 有什么作用？

答：BF16 主要用于接近现代 GPU 大模型训练的常见精度设置。相比 FP32，它能降低计算和显存压力，而且不需要像 FP16 那样引入 GradScaler。

这也影响 checkpoint 语义：因为主线使用 BF16/FP32，所以项目没有实现 FP16 GradScaler state 的保存。如果将来支持 FP16 mixed precision，就应该把 scaler state 纳入 checkpoint。

### Q27. 面试官：activation checkpointing 在这个项目里是什么定位？

答：activation checkpointing 是显存优化手段，通过反向时重算部分激活来降低显存占用。它会牺牲一些计算时间，但能让更大模型或更长序列跑起来。

在本项目里它主要用于 memory pressure 和大模型压力场景，和 FSDP/ZeRO 一起观察显存与吞吐的取舍。

### Q28. 面试官：如果让你继续优化 DDP/FSDP 性能，你会先看哪里？

答：我会先看 profiler 的 step breakdown，确认瓶颈在 forward/backward、optimizer 还是 collective。对 DDP，我会重点看 gradient bucket、accumulation 同步次数、all-reduce 时间和 overlap；对 FSDP，我会看 all-gather/reduce-scatter 调用频率、auto wrap 粒度、prefetch 行为和显存峰值。

然后再结合模型规模判断是否值得改策略。小模型优先减少框架和通信开销，大模型优先确认显存是否安全。

## 四、Checkpoint、Resume 和故障恢复

### Q29. 面试官：你这个 checkpoint 到底保存了哪些状态？

答：保存 model、optimizer、scheduler、TrainState 和每个 rank 的 CPU/CUDA RNG state。TrainState 里有 global step、micro step、tokens seen、seed 和 config fingerprint。

这样做的目标是精确恢复，而不是功能性恢复。功能性恢复只要能继续跑，精确恢复要求后续状态和连续训练一致。

### Q30. 面试官追问：为什么只恢复 global step 不够？

答：global step 只表示训练进度，不表示完整训练状态。optimizer 里的动量、Adam 的一二阶矩、scheduler 的 completed steps、dropout 的 RNG 状态都会影响后续参数更新。

如果只恢复 global step 和模型权重，训练可能能继续跑，但下一步的学习率、随机 mask 或 optimizer update 都可能和连续训练不同。

### Q31. 面试官：READY 文件为什么重要？

答：分布式 checkpoint 是多 rank 写入，目录存在不代表所有 shard、metadata 和 RNG 文件都写完。如果恢复逻辑只看目录名，preemption 发生在保存中间时，就可能读到半成品 checkpoint。

READY 是发布完成的提交标记。只有写完 DCP state、RNG、metadata，并完成必要 barrier 后，rank 0 才写 READY，然后原子替换目录并更新 latest。恢复路径只信任带 READY 的 checkpoint。

### Q32. 面试官追问：latest 指针损坏怎么办？

答：恢复 `latest` 时不能盲信 latest 文件。项目会通过 READY 扫描可用的 `step_*` 目录，跳过没有 READY 的半成品目录。

这能处理一种常见情况：latest 指向了一个看起来新的 checkpoint，但那个 checkpoint 没发布完成。正确行为是回退到最新的 READY checkpoint，而不是冒险加载。

### Q33. 面试官：为什么 checkpoint 要先写临时目录？

答：临时目录隔离“正在写”和“已经发布”两个状态。所有 rank 先写 `.step_xxx.tmp`，写完后 rank 0 才把它原子替换成正式 `step_xxx`。

这样恢复逻辑可以简单可靠：正式目录加 READY 才是可恢复点。没有临时目录的话，恢复进程可能看到一个正在写的正式目录。

### Q34. 面试官：为什么要保存每个 rank 的 RNG？

答：分布式训练里每个 rank 的随机状态可能不同。比如 dropout、activation checkpointing 中的随机路径、未来的数据增强都可能依赖本地 RNG。

如果只保存 rank 0 的 RNG，其他 rank 恢复后的随机序列会漂移。项目按 `rng_state_rank_xxxxx.pt` 保存每个 rank 的 CPU/CUDA RNG，这样才能做精确恢复。

### Q35. 面试官追问：你的 synthetic data 已经 deterministic 了，为什么还要 RNG？

答：synthetic data deterministic 只保证输入 batch 可复现，不保证模型内部随机路径可复现。dropout 的 mask、某些 checkpoint 重算路径或未来扩展的数据增强仍依赖 PyTorch RNG。

所以 data determinism 和 runtime RNG determinism 是两层东西。前者解决 batch 顺序，后者解决训练计算里的随机性。

### Q36. 面试官：`checkpoint verify` 为什么不直接比较文件 hash？

答：文件 hash 比较的是物理存储布局，不一定等价于训练状态。DCP 的 shard 文件、metadata 顺序、创建时间或布局可能不同，但加载后的模型和 optimizer 状态是等价的。

所以项目会加载 checkpoint，重建 model、optimizer、scheduler，再对 state dict、TrainState 和 RNG 做 digest。比较的是训练语义，而不是文件字节。

### Q37. 面试官追问：FSDP 的 checkpoint verify 有什么特别难点？

答：FSDP 的 state dict 里可能出现 ShardedTensor 或 DTensor，不能当普通 tensor reshape。第一次实现如果直接按 tensor 处理，就容易在分片对象上失败。

项目的做法是识别 DTensor 和 local_shards，先按本 rank 的 local shard 计算 digest，再用 `all_gather_object` 按 rank 顺序聚合。这样不用把完整 state all-gather 到单卡，也能比较两份分布式 checkpoint。

### Q38. 面试官：怎么验证 resume 是 exact match？

答：跑两条路径。一条是 continuous training，直接训练到目标 step 并保存 checkpoint。另一条是 interrupted training，先保存中间 checkpoint，再从 latest resume 到同样 step。

然后用 `checkpoint verify` 比较两边最终 checkpoint 的 model、optimizer、scheduler、TrainState 和 RNG digest。只要其中任何一个不一致，就不能说 exact resume。

### Q39. 面试官：旧 checkpoint 兼容你怎么处理？

答：新增字段后，旧 checkpoint 的 config fingerprint 可能不包含这些字段。如果直接用新 fingerprint 比较，旧 checkpoint 会被错误拒绝。

所以项目保留了 legacy fingerprint 逻辑。对于旧 schema 且用户使用默认语义时，可以做功能性恢复；但如果缺少 RNG state，就会标记 `resume_deterministic=false`，不能声称精确恢复。

### Q40. 面试官追问：为什么不强行让旧 checkpoint 也 exact？

答：因为旧 checkpoint 没保存足够状态，无法凭空恢复精确随机状态。强行标成 exact 是错误的。

正确做法是区分 functional resume 和 deterministic resume。能继续训练是一回事，能证明和连续训练一致是另一回事。

### Q41. 面试官：rank crash 实验验证了什么？

答：它验证真实 worker 被 `SIGKILL` 后，torchrun launcher 会非零退出，并且最后一个 READY checkpoint 没有被破坏。然后手工重启，从 READY checkpoint 恢复到目标 step，再做 exact verify。

这里我会明确说恢复模式是 manual restart，不是 TorchElastic 自动恢复。这个边界必须讲清楚。

### Q42. 面试官：如果训练过程中出现 NaN，你怎么处理？

答：项目在 loss 和 grad norm 上做非有限值检测，并通过 all-rank reduction 确保所有 rank 一起 fail-fast。如果任意 rank 发现 NaN 或 Inf，就在 optimizer step 前清空梯度并中止。

这样避免一个 rank 继续 step、另一个 rank 抛错，导致分布式状态分叉。训练系统里一致失败比错误继续更重要。

## 五、Profiler、性能统计和证据可信度

### Q43. 面试官：为什么要把 profiler 做成单独命令？

答：Profiler 会改变 kernel 调度、记录开销、内存行为和 Python 开销。如果默认在 benchmark 里打开，tokens/sec 就不再代表正常训练性能。

所以项目把 `train` 和 `profile` 分开。`train` 用来产生稳定性能数字，`profile` 用来解释慢在哪里。这是性能工程里很重要的分工。

### Q44. 面试官：Profiler 里你主要看哪些指标？

答：先看 step breakdown：data time、forward/backward、optimizer step 和 total step time。然后看每 rank top CUDA/CPU op，以及 collective 相关事件，比如 nccl all-reduce、all-gather、reduce-scatter。

最后看 rank diagnostics，包括 step min/p50/max、straggler ratio 和 collective time per step。这样可以判断瓶颈是计算、通信、optimizer、host overhead 还是 rank 不均衡。

### Q45. 面试官追问：为什么不能只看 `key_averages()` 判断通信计算 overlap？

答：`key_averages()` 聚合的是 op 统计，不保留完整时间线关系。它能告诉我哪些 op 总耗时高，但不能证明这些通信是否和计算在不同 CUDA stream 上重叠。

要判断 overlap，需要看 Chrome trace 或 Perfetto 的时间线。所以项目报告里如果没有 trace 证据，会明确写 overlap 未确定，而不是凭聚合表下结论。

### Q46. 面试官：正式性能结论为什么要 repeat？

答：GPU benchmark 会受到调度、温度、频率、缓存、系统噪声和其他作业影响。单次短跑只能作为 smoke 或覆盖性结果，不能代表稳定结论。

项目使用 repeat=3 时，每个 trial 独立初始化，并报告 mean、std、min、max。这样至少可以看出结果波动，而不是只挑一个好看的数字。

### Q47. 面试官：为什么要记录 provenance？

答：训练 benchmark 对环境非常敏感。PyTorch、CUDA、cuDNN、NCCL、driver、base image、源码 commit、容器 image、启动命令都会影响结果。

如果没有 provenance，别人很难复查这个数字，也很难判断两批结果能不能横向比较。项目里缺 provenance 的旧结果不会被升级成正式证据。

### Q48. 面试官追问：如果 GPU 上有其他进程，会怎么处理？

答：正式性能实验应该拒绝或至少标记环境不干净。比如 Megatron smoke 里检测到 concurrent GPU compute processes，就把 performance_valid 标成 false，Markdown 里不展示吞吐和显存。

这不是保守过度，而是避免把被污染的结果包装成正式 benchmark。性能数字最怕来源不清。

### Q49. 面试官：memory pressure 矩阵说明了什么？

答：它说明不同并行策略在模型规模变化下的价值不同。小模型上 DDP 往往吞吐更好，FSDP/ZeRO 的管理和通信开销更明显；模型变大后，分片策略的显存优势开始变得关键。

项目里 stress 规模下 DDP OOM，而 FSDP 可以完成训练，这个结果说明 FSDP 的核心价值是可训练性。不是所有场景都追求最高 tokens/sec，有时先要能放得下。

### Q50. 面试官：你怎么判断一个 benchmark 结果可信？

答：我会看四点。第一，配置是否固定，包括模型、batch、seq length、precision、warmup 和 steps。第二，统计方法是否合理，比如 CUDA synchronize、跨 rank max、repeat mean/std。第三，环境是否可追溯，包括 commit、image、driver、NCCL。第四，报告是否诚实区分 smoke、正式 benchmark 和无效性能数据。

如果这些条件不满足，我最多把结果当调试信息，不会写成正式结论。

## 六、通信、MoE、Tensor Parallel、ZeRO 和 Megatron

### Q51. 面试官：你为什么要单独做 communication benchmark？

答：训练吞吐只能看到最终结果，看不清通信 primitive 的基础特征。DDP、FSDP、MoE 分别依赖不同 collective：DDP 主要是 all-reduce，FSDP 常见 all-gather 和 reduce-scatter，MoE expert parallel 关键是 all-to-all。

单独测 collective 可以帮助解释训练性能。比如小 tensor 更偏 latency-bound，大 tensor 更能暴露带宽上限；all-to-all uneven split 可以模拟 MoE router 造成的不均衡通信。

### Q52. 面试官：all-reduce、all-gather、reduce-scatter 和 all-to-all 分别对应什么？

答：all-reduce 主要对应 DDP 梯度同步，每个 rank 得到聚合后的完整梯度。all-gather 常用于 FSDP 在计算前收集完整参数。reduce-scatter 常用于 FSDP 把聚合后的梯度分片回各 rank。all-to-all 主要对应 MoE expert parallel 的 token dispatch 和 combine。

这四个 collective 的代价不只取决于带宽，还取决于消息大小、调用频率、rank 数、拓扑和是否能与计算 overlap。

### Q53. 面试官：MoE 为什么重点看 all-to-all，而不是 all-reduce？

答：MoE 的 expert parallel 中，参数不一定像 DDP 那样每步全量同步。核心问题是 router 会把本地 token 分配给不同 expert，而 expert 分布在不同 rank 上，所以 token 要跨 rank 重新分发。

这个过程就是 all-to-all。真实 MoE 还会有 capacity、overflow、load imbalance、token packing 和 combine 成本，所以只看 all-reduce 不能代表 MoE 通信压力。

### Q54. 面试官追问：为什么要测 uneven split？

答：真实 MoE 里 router 不会保证每个 rank 发给每个 peer 的 token 数完全一样。某些 expert 可能更热门，导致部分 rank 收发更多 token。

equal split 可以看链路上限，uneven split 更接近负载不均场景。两者一起看，才能更接近 MoE 的真实通信形态。

### Q55. 面试官：toy Tensor Parallel check 验证了什么？

答：它验证 ColumnParallelLinear 和 RowParallelLinear 的 forward/backward 是否和单卡 reference 一致。Column parallel 按输出维切分，每个 rank 计算一段输出；Row parallel 按输入维切分，每个 rank 计算 partial output 后做聚合。

这个 demo 的重点不是性能，而是切分语义正确。面试里我会先讲清楚切在哪一维、激活怎么流、梯度怎么聚合，再讲实现细节。

### Q56. 面试官追问：为什么 TP MLP 通常是 column 后接 row？

答：Transformer MLP 第一层通常把 hidden 扩到更大的 intermediate 维度，适合按输出维做 column split，让每个 rank 持有一部分 intermediate。第二层再按输入维 row split，把 partial output 聚合回 hidden 维。

这样中间大维度激活可以保持分片，避免不必要的 all-gather。Megatron-style TP 的核心就是让切分后的 forward 和 backward collective 闭环。

### Q57. 面试官：Sequence Parallel 解决什么问题？

答：Sequence Parallel 通常和 Tensor Parallel 配合，把部分 activation 沿 sequence 维切分，降低长序列训练的激活显存压力。

它不是免费优化，会引入额外 collective，也要注意 dropout 等随机路径在分片上的一致性。本项目做的是 toy correctness demo，不声称实现了完整生产级 SP runtime。

### Q58. 面试官：DeepSpeed ZeRO 为什么做成独立 adapter？

答：DeepSpeed Engine 会接管 backward、step、gradient accumulation、optimizer 和 checkpoint 生命周期。如果强行塞进当前 DDP/FSDP `Trainer`，两套状态机容易混在一起。

所以项目把 ZeRO-2/ZeRO-3 做成独立 benchmark adapter，输出归一化 JSON，报告层可以和 DDP/FSDP 横向比较，但核心 runtime 的 checkpoint/resume 语义保持清晰。

### Q59. 面试官：ZeRO-2、ZeRO-3 和 FSDP 怎么对比？

答：ZeRO-2 主要分片 optimizer state 和 gradient，参数仍相对完整；ZeRO-3 连参数也分片。FSDP 也是参数、梯度、optimizer state 分片，但它在 PyTorch runtime 内管理模型包装、all-gather、reduce-scatter 和 state dict。

它们的取舍要结合模型规模和实现栈看。小模型上额外管理成本可能不划算，大模型或显存压力场景下分片能决定能不能训练。

### Q60. 面试官：Megatron-LM 相关你到底做了什么，没做什么？

答：我没有在这个仓库里复刻完整 Megatron runtime。项目内部实现了可验证的 toy TP/SP 和训练 runtime contract；外部固定 Megatron-LM `core_v0.18.2`，跑通了五组 8 卡 TP/PP/DP compatibility smoke，并做了工程 case study。

但这批 Megatron 运行使用的是官方 PyTorch fallback，且 GPU 环境非独占，所以 performance_valid 是 false。它能证明拓扑和启动兼容性，不能当正式性能 benchmark。正式性能结论需要 NGC、独占 GPU、repeat=3 和完整 metadata。

## 七、项目边界和面试收束

### Q61. 面试官：这个项目最大的限制是什么？

答：主要限制是单节点、synthetic data、pretraining-focused。它没有完整多机训练平台，没有真实数据 IO pipeline，没有 RLHF/GRPO，没有推理服务，也没有完整 Megatron 级别的 PP/SP runtime。

但这些限制是明确写进项目边界的。我不会把 toy correctness、compatibility smoke 或单节点结果包装成生产级能力。

### Q62. 面试官：如果继续做，你会优先补什么？

答：我会优先补两类。第一是更正式的 Megatron 性能矩阵：NGC 环境、独占 GPU、repeat=3、完整 provenance。第二是多机 NCCL 证据：固定 rendezvous、网卡配置、跨节点 collective 和故障诊断。

如果从 runtime 角度继续，则会考虑更完整的 reshardable checkpoint、TorchElastic 自动恢复、FP16 scaler state 和真实 dataloader pipeline。

### Q63. 面试官：你怎么把这个项目放到简历里？

答：我会写成：实现 Docker 化最小 PyTorch 分布式训练 runtime，支持 DDP/FSDP/DeepSpeed ZeRO、BF16、gradient accumulation、activation checkpointing、分布式 checkpoint/resume、Profiler、collective benchmark、MoE all-to-all 和 toy Tensor Parallel correctness；在 8x A100 上完成吞吐、显存、repeat 统计和故障恢复证据，并记录完整 provenance。

这句话既覆盖技术点，也没有夸大成完整生产框架。

### Q64. 面试官：你最想让面试官记住哪一点？

答：我最想让面试官记住：这个项目的核心不是“我跑了几个 benchmark”，而是“我把训练系统的关键状态和边界做成了可验证证据”。

比如 DDP/FSDP 不只是概念对比，而是有吞吐、显存、profiler 和 memory pressure；checkpoint 不只是能加载，而是有 READY、RNG、scheduler 和 exact verify；Megatron 不只是说懂，而是明确区分 toy correctness、compatibility smoke 和正式性能 benchmark。

### Q65. 面试官：最后用一句话总结你的工程判断。

答：我的工程判断是先把训练 runtime 的核心 contract 做小、做准、做可复查，再谈扩展。

这个项目里我刻意控制边界，用 synthetic data 固定变量，用 strategy 隔离 DDP/FSDP，用 READY 和 fingerprint 保证 checkpoint 语义，用 profiler 和 provenance 解释性能，用故障 smoke 验证恢复边界。这样项目虽然不是大而全，但每个结论都能落到代码和证据上。

## 八、投递和通用面经

这一节更偏投递场景，问题会更像真实面试里的开场、自我介绍、岗位匹配和通用训推 Infra 高频题。

### 8.1 投递表达

### Q66. 面试官：请你做一个 30 秒自我介绍。

答：我主要做的是训练 Infra 和分布式 runtime 方向，核心项目是 MiniTrainBench。这个项目围绕 DDP、FSDP、ZeRO、checkpoint/resume、Profiler、通信 microbenchmark 和故障恢复做了一条完整证据链，重点验证训练系统的状态、性能和边界。

如果对方继续听，我会补一句：我不是只跑 benchmark，而是把 checkpoint 原子发布、RNG 精确恢复、repeat 统计和 provenance 记录都做了，能把训练系统的行为讲清楚、证据留住。

### Q67. 面试官：如果给你 90 秒，你怎么介绍这个项目？

答：我会先说目标。MiniTrainBench 是一个最小分布式训练 runtime，用 synthetic token 驱动 GPT-like 模型，目的是把训练系统里最容易出问题的地方做成可验证证据：DDP/FSDP/ZeRO 的通信和显存差异、gradient accumulation 的同步语义、checkpoint/resume 的精确恢复、Profiler 的瓶颈定位、MoE all-to-all 的通信形态。

然后我会说结果。项目已经在 8x A100 上做了 1/2/4/8 卡 DDP/FSDP、collective benchmark、memory pressure 和真实 rank crash 验证。最后我会补边界：它是单节点、pretraining-focused，不是假装成完整生产训练框架，但每个能力都有源码和结果证据。

### Q68. 面试官：为什么你觉得自己适合训推 Infra？

答：因为我做项目时关注的不是模型效果，而是训练和推理系统里真正会卡住业务的东西：状态管理、性能定位、通信、显存、故障恢复和可复现性。训推 Infra 的工作本质上就是把“系统跑得稳、跑得快、出问题能查、恢复能对”这几件事做实。

我的项目虽然以训练为主，但已经覆盖了很多共通能力：分布式通信、性能剖析、checkpoint 语义、故障边界和证据链。这些能力迁移到推理系统里，通常对应的是调度、KV cache、批处理、延迟和 SLA 管理。

### Q69. 面试官：如果只能让你讲一个最强的点，你会讲什么？

答：我会讲精确 checkpoint/resume。这个点最能体现我不是只会“跑通”，而是能把训练状态讲清楚、把恢复语义做对。

我保存的不只是模型权重，而是 model、optimizer、scheduler、TrainState 和每 rank RNG；恢复时不是只看目录在不在，而是检查 READY、fingerprint、world size 和 state digest；验证时还会把连续训练和中断恢复做 exact match。这个点能很好地说明我对训练系统状态的理解。

### Q70. 面试官：你会怎么讲这个项目的边界？

答：我会直接讲清楚：它是单节点、pretraining-focused、synthetic data 驱动的最小 runtime，不是完整多机平台，不是 RLHF/GRPO，也不是推理服务框架。Megatron 只做 compatibility smoke，没有把正式性能结果和 smoke 混在一起。

这样讲的好处是，面试官会知道我对项目边界是诚实的，不会把没有做完的能力包装成已经落地的系统。训练 Infra 面试里，这种边界意识通常比“我做了很多功能”更重要。

### Q71. 面试官：如果面试官追问你“有没有做过线上推理”，你怎么答？

答：我会如实说当前项目的主线是训练 Infra，还没有把线上推理服务做成完整系统。但我会补充我已经具备的通用能力：对吞吐、延迟、显存、调度、通信和瓶颈定位都有系统化理解，这些是推理 Infra 的底层能力。

如果对方继续问，我会把话题拉回到可迁移点：训练和推理虽然目标不同，但都离不开资源调度、观测、故障处理和性能优化。我的项目能说明我有这类系统思维。

### Q72. 面试官：为什么你没有去做算法岗，而是偏 Infra？

答：因为我更关心的是模型跑起来的系统条件，而不是单纯追指标。训练和推理里很多难点不是“模型结构怎么改”，而是“怎么让系统稳定、可复现、可扩展地运行”。这个方向更接近我的兴趣和项目投入方式。

另外，Infra 工作会同时碰到性能、工程、分布式、故障处理和系统设计，这些问题比较符合我在这个项目里积累的能力。

### Q73. 面试官：面试最后你会反问什么？

答：我会问三个方向。第一，这个岗位当前最痛的点是什么，是训练吞吐、稳定性、显存还是推理延迟。第二，团队现有的分布式栈是什么，比如 PyTorch、FSDP、DeepSpeed、vLLM、Triton 或自研调度。第三，岗位更看重系统设计、性能优化还是稳定性治理。

这样问的目的不是套话，而是确认自己做的事情和团队真实需求是否对齐。

### 8.2 训练 Infra 高频追问

### Q74. 面试官：训练平台里，你排查问题会先看什么？

答：我一般先看三件事：日志、指标和状态。日志告诉我报错发生在什么阶段，指标告诉我是不是吞吐、显存或通信异常，状态告诉我 checkpoint、scheduler、RNG 和进程组是否一致。

如果是分布式训练，我会额外看 rank 间是否一致。很多问题不是“算法错了”，而是某个 rank 慢了、坏了，或者状态不一致导致的。

### Q75. 面试官：如果训练 OOM，你怎么排查？

答：我会按从便宜到昂贵的顺序排查。先看 batch size、seq length、模型宽度和 activation checkpointing，再看是不是 optimizer state、gradient accumulation 或 FSDP/ZeRO 带来的峰值开销，最后看是否是临时 all-gather 或内存碎片导致的尖峰。

如果是大模型训练，我会特别注意显存峰值不一定出现在 forward 的主计算段，也可能出现在参数聚合、梯度同步或者保存 checkpoint 的时候。这个项目的 memory pressure 矩阵就是在验证这类场景。

### Q76. 面试官：如果训练 step time 抖动很大，你怎么查？

答：先区分是数据抖动、通信抖动还是计算抖动。数据抖动通常看 data time；通信抖动看 NCCL、collective 和 rank straggler；计算抖动则看 kernel、显存分配和 CPU 开销。

我一般会把同一批实验 repeat 多次，再看 p50、p95 和 max。如果只有少数 step 异常高，那更像偶发性资源干扰；如果持续抖动，那多半是调度、通信或者数据路径的问题。

### Q77. 面试官：通信慢了，你怎么定位是 NCCL、拓扑还是代码问题？

答：先看通信类型和消息大小，再看是否有 overlap，最后看硬件和拓扑。比如 all-reduce 慢，可能是 bucket 太小、调用太频繁，也可能是链路带宽不够或 rank 间不均衡。

如果是多机，还要看网卡配置、rendezvous、NCCL 环境变量和节点间连通性。我的判断原则是：先证明是通信慢，再区分是算法问题、代码组织问题还是环境问题。

### Q78. 面试官：如果 checkpoint 恢复后 loss 曲线不一样，你会怎么查？

答：我会先查 scheduler、optimizer state 和 RNG。因为这三类状态最容易导致“能恢复但不等价”。如果这三者都一致，再看数据顺序、dropout、activation checkpointing 和 batch 生成逻辑。

这个项目里我专门做了 exact verify，就是为了把“能恢复”拆成“能跑”和“能证明一致”。如果恢复后曲线不一样，说明至少有一类状态没有被正确恢复。

### Q79. 面试官：你会怎么做一次训练性能优化实验？

答：我会先定义单一目标，比如把 step time 降低 10%，或者把显存下降 20%。然后固定模型、batch、精度、seed 和环境，只改一个变量，跑 repeat 并记录 mean/std。

最后我会同时看吞吐、显存、通信和 profile，避免只赢一个数字却把另一个指标搞坏。训练优化不是“改一个参数就好”，而是要对权衡关系有数。

### Q80. 面试官：DDP、FSDP、ZeRO 你会怎么选？

答：如果模型不大、显存够、目标是简单稳定和高吞吐，我会先看 DDP。 如果显存开始紧张，但仍希望保留 PyTorch 原生训练语义，我会考虑 FSDP。 如果已经在 DeepSpeed 栈里，或者希望用 ZeRO 的分片策略对齐特定训练模式，我会考虑 ZeRO。

我的判断不是“谁高级用谁”，而是看模型大小、显存余量、通信成本、工程复杂度和团队栈。

### Q81. 面试官：怎样判断一个训练优化是安全的？

答：至少要看三层。第一，数值上训练还能收敛，没有引入 NaN 或明显漂移。第二，状态上 checkpoint、resume 和 scheduler 仍然正确。第三，实验上 repeat 统计有支撑，不是单次偶然赢了。

如果优化只让某一次 tokens/sec 变高，但 checkpoint 不能恢复、RNG 不一致或者某个 rank 更容易出错，我会认为这个优化不安全。

### 8.3 推理 Infra 高频追问

### Q82. 面试官：prefill 和 decode 有什么区别？

答：prefill 是把输入 prompt 整体灌进去，主要是建立初始上下文；decode 是在已有上下文基础上一个 token 一个 token 地生成。通常 prefill 更偏计算密集，decode 更偏时延敏感。

推理系统里这两段的优化目标不一样。prefill 更关心吞吐和并行度，decode 更关心单请求延迟和尾延迟。

### Q83. 面试官：KV cache 是什么，为什么重要？

答：KV cache 是把 attention 里每层的 key/value 缓存下来，避免每生成一个新 token 都重新算历史上下文。它能显著减少 decode 阶段的重复计算。

代价是显存占用会随上下文长度和并发增长，所以推理系统里 KV cache 管理是核心问题之一。你既要让它快，又要控制内存别爆。

### Q84. 面试官：batching 和 continuous batching 有什么区别？

答：batching 是把多个请求凑成一个 batch 再一起跑，吞吐通常更好。continuous batching 则允许新请求在运行中插入，不需要等一个 batch 全部结束，更适合在线服务。

前者更像离线高吞吐场景，后者更像在线推理服务。面试里我会强调：推理系统的难点不是“能不能 batch”，而是如何在吞吐、延迟和公平性之间做动态平衡。

### Q85. 面试官：为什么推理里总在讲 latency、throughput 和 SLA？

答：因为推理和训练的目标不同。训练更关注整体吞吐和扩展效率，推理更关注用户感知的响应速度，尤其是 P50、P95、P99 延迟和 SLA 是否被满足。

很多推理优化会牺牲一部分吞吐换更低延迟，或者反过来。所以推理 infra 面试里一定要讲清楚：你的优化是在服务什么目标，不是单纯把数字做大。

### Q86. 面试官：量化在推理里解决什么问题？

答：量化主要是降低模型参数和中间激活的存储与计算成本，通常能帮助提升吞吐、降低显存或让更大的模型能部署。

但量化会带来精度损失和工程复杂度。面试里不要只说“量化更快”，而要说清楚它可能影响输出质量、调试难度和硬件兼容性。

### Q87. 面试官：如果让你设计一个推理服务，你会先想哪些模块？

答：我会先想请求路由、调度器、模型 worker、KV cache 管理、batching 策略、指标监控和故障恢复。

然后再看扩缩容、权重加载、版本切换、限流和降级。推理系统不是只把模型挂上去，而是要把请求生命周期、资源管理和 SLA 管起来。

### Q88. 面试官：训练 infra 和推理 infra 的共同点和差异是什么？

答：共同点是都很看重分布式、显存、通信、调度、观测和故障处理。差异是训练更关注长期状态、optimizer 和 checkpoint，推理更关注请求排队、KV cache、batching 和尾延迟。

所以我会说，训练 infra 和推理 infra 是同一类系统问题的两个分支。一个关注“怎么把模型学出来”，一个关注“怎么把模型稳定地服务出去”。

### Q89. 面试官：如果你去投推理 Infra，这个项目怎么讲才不违和？

答：我会把它讲成一个“训练侧的系统基础能力证明”，而不是假装它已经是推理系统。重点讲我对分布式、性能、状态、观测和故障恢复的理解，这些是训练和推理都共享的底层能力。

然后我会主动补一句：我虽然主线在训练，但我已经清楚推理系统的核心矛盾，比如 prefill/decode、KV cache、batching 和 P99 latency。这样讲既诚实，也能让面试官知道我不是只会训练那一侧。

## 九、训练 Infra 系统设计题

这一节更像系统设计面试。答题时不要一上来堆组件名，要先讲目标，再讲关键链路，最后讲边界。

### Q90. 面试官：如果让你设计一个分布式训练平台，你会怎么拆？

答：我会先把它拆成五层：作业入口、资源调度、训练执行、状态持久化和观测诊断。作业入口负责接收训练配置、镜像、代码版本和资源需求；资源调度负责 GPU、节点、网络和队列；训练执行负责 launcher、rank 环境、容器和日志；状态持久化负责 checkpoint、artifact 和 provenance；观测诊断负责指标、trace、告警和失败归因。

如果结合 MiniTrainBench，我会说它实现的是训练执行和状态验证这一层的最小切片：Trainer、strategy、checkpoint、profiler、doctor 和结果报告。它不是完整平台，但能证明训练 runtime 的核心 contract。

### Q91. 面试官追问：一个训练 job 的生命周期你怎么设计？

答：我会按 submit、preflight、schedule、launch、run、checkpoint、monitor、finish/recover 这条链路设计。submit 阶段校验配置，preflight 阶段检查镜像和环境，schedule 阶段分配 GPU 和节点，launch 阶段注入 rank 环境，run 阶段持续采集指标，checkpoint 阶段发布可恢复状态，最后根据成功、失败或抢占决定收尾或恢复。

关键是每个阶段都要有明确状态。比如 checkpoint 不能只看目录存在，必须有 READY；性能结果不能只看 JSON 存在，必须有 provenance；失败不能只记录 exit code，还要知道是 launcher、runtime、collective、checkpoint 还是环境问题。

### Q92. 面试官：训练平台的资源调度最难的点是什么？

答：难点不只是“分配几张卡”，而是如何保证资源满足训练拓扑。单机多卡主要看 GPU 数和显存，多机还要看节点数量、网络拓扑、IB/RDMA、NCCL 可达性、容器网络和 rendezvous。

如果调度只知道 GPU count，不知道网络和拓扑，就可能把一个需要高速互联的 job 调到跨慢链路节点上，最后表现成 NCCL 慢或 hang。训练 Infra 里调度和通信不是独立问题。

### Q93. 面试官：checkpoint 服务应该怎么设计？

答：我会把 checkpoint 服务设计成“写入、发布、发现、清理、校验”五个环节。写入阶段可以允许临时状态，发布阶段必须有提交标记，发现阶段只返回已发布 checkpoint，清理阶段不能删除唯一可恢复点，校验阶段要能比较训练语义而不是只比较文件。

MiniTrainBench 里的临时目录、READY、latest、keep-last 和 checkpoint verify 就是这个设计的最小实现。面试里可以强调：checkpoint 不是存储问题，而是训练状态一致性问题。

### Q94. 面试官追问：为什么 checkpoint 不能直接交给对象存储目录管理？

答：对象存储能解决持久化，但不能自动解决训练语义。多 rank 写 checkpoint 时，某些 shard 可能已经写完，另一些还没写完；对象存储里看到目录或前缀存在，不代表 checkpoint 已经可恢复。

所以平台层仍然需要发布协议，比如 manifest、READY、latest 和元数据校验。对象存储负责可靠存放，训练平台负责定义“什么时候算可恢复”。

### Q95. 面试官：训练平台的观测系统应该采什么指标？

答：我会分成资源、训练、通信和状态四类。资源指标包括 GPU 利用率、显存、温度、功耗、网络；训练指标包括 loss、lr、grad norm、step time、tokens/sec；通信指标包括 collective 时间、调用次数、rank spread；状态指标包括 checkpoint 进度、latest、resume path、失败类型和 provenance。

只采 GPU utilization 不够。一个 job GPU 利用率高但 loss 是 NaN，或者 step time 正常但 checkpoint 不可恢复，都不是健康训练。

### Q96. 面试官：日志、metrics、trace 三者怎么分工？

答：日志用于回答“发生了什么”，metrics 用于回答“趋势是否异常”，trace 用于回答“一步里面时间花在哪里”。三者不能互相替代。

比如 step time 变慢，metrics 会告诉我变慢了，日志可能告诉我是不是重启或 checkpoint，trace 才能告诉我是 NCCL、optimizer 还是某个 CUDA op 变慢。MiniTrainBench 把 benchmark 和 profiler 拆开，就是为了让稳定数值和定位证据各司其职。

### Q97. 面试官：从单机扩到多机，最先要补哪些能力？

答：首先要补 rendezvous、网络和环境一致性。多机训练比单机更容易在启动阶段出问题，比如 MASTER_ADDR/PORT 不通、rank/world size 不一致、NCCL_SOCKET_IFNAME 选错、CUDA/NCCL 版本不一致、容器网络不通。

其次要补多机 checkpoint 发现和故障恢复。单机上路径共享比较简单，多机上 checkpoint 存储、权限、带宽和一致性都要重新设计。当前项目有 doctor 和多机模板，但没有正式多机 benchmark，所以面试时要明确边界。

### Q98. 面试官：你会怎么设计 preflight 检查？

答：preflight 应该在真正训练前失败，而不是让 job 跑半小时后才失败。我会检查镜像 digest、代码 revision、GPU 数、driver/CUDA/NCCL、网络接口、rendezvous 端口、磁盘空间、对象存储权限和当前 GPU 是否被占用。

本项目的 `doctor` 和 provenance gate 就是这个方向的轻量版本。正式训练平台还会把 preflight 结果写入 job metadata，方便失败时归因。

### Q99. 面试官：训练平台怎么处理抢占和恢复？

答：抢占恢复的核心是“恢复点不能被污染”。平台应该定期保存 checkpoint，抢占时尽量触发最后一次保存；如果来不及保存，也必须保证 latest 指向的还是上一个 READY checkpoint。恢复时根据 job config、world size、strategy 和 checkpoint metadata 判断能否恢复。

MiniTrainBench 验证的是 manual restart：rank crash 后最后 READY checkpoint 不变，再手工重启 exact verify。它没有实现调度器自动拉起，但恢复语义已经有证据。

### Q100. 面试官：你会怎么设计训练平台的结果发布？

答：我会把结果分成 raw evidence、validated result 和 public report 三层。raw evidence 保存原始 JSON、命令、日志、环境；validated result 要通过 provenance、repeat、环境独占和 schema 校验；public report 只展示可比较的结果。

这个设计能避免把 smoke、调试结果和正式 benchmark 混在一起。MiniTrainBench 对 Megatron smoke 的处理就是例子：compatibility 成功不等于 performance valid。

### Q101. 面试官：训练平台里为什么要强调 schema？

答：因为不同 adapter 和 runtime 会产生不同字段。如果报告层假设所有结果都有相同字段，就容易因为一个 ZeRO adapter 缺 `grad_norm` 之类字段直接崩掉，或者更糟，把缺失字段当成 0。

正确做法是明确必填字段和可选字段。缺失字段要显示 unknown 或 `-`，不能静默编造。性能平台的 schema 也是训练 Infra 的一部分。

### Q102. 面试官：如果让你做 checkpoint service 的 SLO，你会定义什么？

答：我会定义保存成功率、恢复成功率、平均保存耗时、P95 保存耗时、latest 可用性、恢复校验通过率和过期 checkpoint 清理延迟。只说“能保存”不够，因为 checkpoint 还会影响训练效率和故障恢复时间。

另外我会把“功能恢复”和“精确恢复”分开统计。前者表示能继续训练，后者表示状态能和连续训练一致。训练平台应该知道自己提供的是哪一种保证。

### Q103. 面试官：这个项目如果接入真实训练平台，最需要补什么？

答：我会优先补三块。第一是多机和调度器集成，包括 rendezvous、hostfile、节点健康检查。第二是远端 checkpoint 存储，包括对象存储、manifest 和多机可见路径。第三是更完整的观测和告警，包括 rank 级指标、trace 采样和失败分类。

但我不会一上来就把所有平台能力塞进 MiniTrainBench。它的价值是作为 runtime contract 和 benchmark evidence 的最小基准，接入平台时应该保持这个边界。

## 十、训练 Infra 故障排查题

这一节统一按“现象 -> 定位 -> 根因 -> 修复 -> 验证 -> 边界”答。面试时这种结构比直接猜原因更稳。

### Q104. 面试官：训练 hang 住了，你怎么排查？

答：现象是训练不退出，也没有继续打 step 日志。定位时我会先看是不是所有 rank 都 hang，还是只有某个 rank 退出或卡住；再看最后一条日志停在 data、forward、backward、optimizer、checkpoint 还是 collective。

根因常见有 rank 没全部进入 collective、world size 配错、NCCL 网络问题、某个 rank OOM 后其他 rank 等待、或者 checkpoint barrier 不一致。修复要针对根因：rank mismatch 修 launcher，NCCL 修网络和环境，OOM 修显存，barrier 不一致修控制流。验证不是“重新跑一次没挂”，而是加最小复现、rank 日志和 timeout 保护。

### Q105. 面试官追问：hang 和慢怎么区分？

答：慢是 step 仍然推进，只是 step time 变大；hang 是长时间没有进度，通常 stuck 在某个同步点。区分方法是看 global_step 是否变化、GPU 是否还有 kernel、NCCL 是否有活跃通信、日志是否持续输出。

如果是慢，就进入性能定位；如果是 hang，就进入同步和进程状态定位。把慢误判成 hang，可能会去查错误方向；把 hang 当成慢，可能浪费很多时间等一个不会恢复的 job。

### Q106. 面试官：NCCL 报错或 timeout，你会怎么定位？

答：我会先确认 rank/world size、MASTER_ADDR/PORT 和节点连通性，再看 NCCL_SOCKET_IFNAME、IB/GDR、CUDA/NCCL 版本和容器网络。接着用小的 collective benchmark 验证通信路径，最后再跑训练。

根因可能是网络不可达、网卡选错、版本不一致、某个 rank 提前退出或 tensor shape 不一致。修复后必须用 doctor、collective benchmark 和训练 step 三层验证，不能只看训练命令偶然跑过。

### Q107. 面试官：出现 NaN，你怎么查？

答：现象是 loss 或 grad norm 变成 NaN/Inf。定位时先看 NaN 出现在 forward、backward 还是 optimizer step 后；再看 learning rate、precision、grad norm、输入数据和最近一次 checkpoint。

根因可能是学习率过大、混合精度溢出、梯度爆炸、数据异常或某个 op 数值不稳定。修复可能是降低 LR、加 grad clipping、检查 BF16/FP16 配置、定位异常 batch。验证要看所有 rank 都 fail-fast，不能让部分 rank 继续 optimizer step。

### Q108. 面试官：OOM 发生在训练中途，不是第一步，你怎么查？

答：中途 OOM 往往不是模型静态放不下，而是某些阶段出现峰值。定位时我会看 OOM 发生在 forward、backward、optimizer、checkpoint、eval 还是某个 all-gather 阶段，并记录峰值显存曲线。

根因可能是序列长度变化、activation checkpoint 未生效、FSDP all-gather 峰值、optimizer state 初始化、内存碎片或外部进程占用。修复要么降低峰值，要么改变策略，比如 FSDP/ZeRO、activation checkpointing、减小 batch 或限制并发。验证要用 memory pressure 或至少多步复现，不只跑一步。

### Q109. 面试官：checkpoint 恢复失败，你怎么排查？

答：先看失败类型：是找不到 checkpoint、缺 READY、metadata 不匹配、DCP load 失败，还是恢复后状态不一致。然后检查 latest 指针、step 目录、metadata、world size、strategy、precision、config fingerprint 和每 rank RNG 文件。

根因可能是半成品 checkpoint、配置变了、world size 变了、旧 checkpoint 缺状态，或者 shard 布局不能直接加载。修复原则是宁可拒绝恢复，也不要错误恢复。验证要用 checkpoint verify，而不是只看命令能继续跑。

### Q110. 面试官：恢复后 loss 曲线漂移，你怎么查？

答：我会先比较恢复点前后的 model、optimizer、scheduler、TrainState 和 RNG。然后看数据顺序是否一致，dropout 是否恢复，gradient accumulation 和 sync mode 是否变化。

根因通常是状态不完整。只恢复 model 和 global step 不够，AdamW state、scheduler completed steps、CUDA RNG 都会影响后续轨迹。修复后要跑 continuous vs interrupted，并做 exact checkpoint verify。

### Q111. 面试官：训练突然变慢但没有报错，你怎么查？

答：先用 metrics 判断是所有 step 慢了，还是偶发 spike。再用 profiler 拆 data、forward/backward、optimizer 和 collective。最后看 rank spread，确认是不是某个 rank 成为 straggler。

根因可能是外部 GPU 进程、GPU 降频、通信拥塞、数据路径抖动、checkpoint 保存变慢或 profiler 被误开启。修复后要用 repeat 和 provenance 证明不是偶然恢复。

### Q112. 面试官：某个 rank 提前退出，其他 rank 卡住，怎么办？

答：这是分布式训练常见问题。定位时先看退出 rank 的日志和 exit code，再看其他 rank 卡在哪个 collective。很多时候真正错误在第一个退出的 rank，其他 rank 只是被动等待。

修复要让失败传播一致，比如 all-rank finite check、launcher 捕获非零退出、设置合理 timeout。验证时要确认所有 rank 都能退出，并且 READY checkpoint 不被失败污染。

### Q113. 面试官：性能回退了 10%，你怎么判断是不是代码问题？

答：先确认对比条件一致：模型、batch、seq、precision、steps、warmup、GPU 独占、driver、image、commit。再看 repeat 的 mean/std，如果方差本来就接近 10%，不能立刻说是代码回退。

如果条件一致，我会用 profiler 对比 step breakdown 和 collective。根因可能是代码路径变化、bucket 行为变化、通信次数增加、checkpoint 或 logging 开销增加。验证要能定位到某个阶段变慢，而不是只看到 tokens/sec 下降。

### Q114. 面试官：多机启动失败，你会先看模型代码吗？

答：不会。我会先看环境和 rendezvous。多机启动失败最常见的问题是 MASTER_ADDR/PORT、hostfile、rank 分配、网络接口、容器网络、NCCL 版本和权限，不一定和模型有关。

只有当 doctor、collective benchmark 和最小 torchrun 都正常后，我才会看训练 runtime。定位顺序错了，很容易把基础设施问题误判成模型问题。

### Q115. 面试官：报告生成失败也算训练 Infra 问题吗？

答：算。训练 Infra 不只负责训练进程，还负责结果可读、可审计和可比较。如果 adapter 输出 schema 变了，报告层崩掉或者把未知字段显示成 0，就会污染结论。

MiniTrainBench 里 ZeRO adapter 缺字段导致报告问题，就是报告 contract 的例子。修复不是给单个 JSON 打补丁，而是明确核心字段和可选字段。

### Q116. 面试官：如果 GPU 被别人占了，实验还能用吗？

答：如果是正式性能实验，不能直接用。外部进程会污染 step time、显存和 NCCL 结果。最多可以保留为 compatibility smoke，不能当 performance benchmark。

正确做法是记录 GPU occupancy，把 performance_valid 标成 false，或者直接拒绝发布。这个项目在 Megatron smoke 里就是这么处理的。

### Q117. 面试官：你怎么给一个新故障写复盘？

答：我会按现象、诊断、根因、修复、验证、边界六项写。现象说明用户看到什么，诊断说明我收集了哪些证据，根因说明属于哪个层级，修复说明改了什么，验证说明怎么证明修复有效，边界说明哪些没有证明。

这种复盘方式能避免“拍脑袋修好了”。训练 Infra 的经验不是记住很多错误，而是建立稳定的排查路径。

## 十一、训练 Infra 性能优化题

性能题不要只说“提高吞吐”。要说明目标指标、影响范围、验证方法和副作用。

### Q118. 面试官：你会怎么做一次训练性能优化？

答：我会先定目标，比如 tokens/sec 提升、step time 降低或显存降低。然后固定实验条件，只改一个变量，跑 repeat，记录 mean/std、显存和 profiler。最后判断优化是否稳定、是否牺牲恢复能力或数值稳定性。

如果一个优化只在单次实验里变快，但方差大、checkpoint 失效或 NaN 风险增加，我不会把它算作安全优化。

### Q119. 面试官：吞吐低，你会先看什么？

答：先看 step breakdown：data、forward/backward、optimizer、communication。不同瓶颈对应不同优化。data 慢看数据管线，forward/backward 慢看模型和 kernel，optimizer 慢看优化器和 ZeRO，communication 慢看 collective 和 overlap。

不能一上来就调 batch size 或换策略。性能优化第一步永远是定位瓶颈，而不是猜参数。

### Q120. 面试官：显存太高，你会怎么优化？

答：先判断显存主要来自参数、梯度、optimizer state、activation 还是通信峰值。参数和 optimizer state 大，可以考虑 FSDP/ZeRO；activation 大，可以考虑 activation checkpointing、减小 seq length 或 batch；通信峰值大，要看 FSDP all-gather 和 accumulation window。

优化显存时要同时看吞吐，因为省显存往往会增加通信或重算。FSDP 的价值不是永远更快，而是在显存紧张时把模型跑起来。

### Q121. 面试官：gradient accumulation 对性能有什么影响？

答：它能用更小显存模拟更大 batch，但会改变通信和计算节奏。DDP 中如果正确使用 `no_sync()`，可以减少一个 optimizer step 内重复 all-reduce；但如果每个 micro-batch 都同步，通信成本不会降。

FSDP 里 accumulation 的取舍更复杂，no_sync 可能保留未分片梯度，降低通信但增加峰值显存。所以不同 strategy 的默认行为不能一样。

### Q122. 面试官：通信和计算 overlap 怎么优化？

答：先要确认是否真的有 overlap，不能只看 key_averages。要看 trace 时间线，确认通信是否和 backward compute 并行。然后再考虑 bucket size、参数顺序、DDP/FSDP prefetch、reduce-scatter timing 等。

如果 trace 证明通信完全串行，才谈 overlap 优化。没有 trace 的时候，我只会说“可能存在 overlap 空间”，不会把理论说成观察结果。

### Q123. 面试官：bucket size 怎么影响 DDP 性能？

答：bucket 太小会导致 collective 调用太频繁，latency 开销高；bucket 太大可能延迟通信启动，减少 overlap。合适的 bucket size 要结合模型层大小、网络带宽和 backward 顺序。

面试里我不会给一个固定数值，因为它依赖模型和硬件。正确回答是先 profile，再调参，再比较 repeat 结果。

### Q124. 面试官：mixed precision 为什么能提升性能？

答：mixed precision 可以降低显存和内存带宽压力，并利用 GPU 的低精度计算单元。BF16 常用于大模型训练，因为数值范围接近 FP32，不像 FP16 那样强依赖 GradScaler。

但它不是无风险优化。不同 op 的数值稳定性、loss scale、optimizer state 和 checkpoint 都要考虑。本项目主线使用 BF16/FP32，所以没有把 FP16 GradScaler state 纳入 checkpoint；如果扩展 FP16，就必须补这个状态。

### Q125. 面试官：DDP、FSDP、ZeRO 的性能优化重点分别是什么？

答：DDP 重点是减少不必要 all-reduce、优化 bucket 和保持高计算通信比例。FSDP 重点是控制 all-gather/reduce-scatter 开销、auto wrap 粒度、prefetch 和显存峰值。ZeRO 重点是 engine 配置、stage 选择、bucket、overlap_comm 和 optimizer state 分片成本。

所以不能用同一套优化思路套所有策略。先明确瓶颈，再选择策略和参数。

### Q126. 面试官：为什么小模型 DDP 快，大模型 FSDP 更有价值？

答：小模型计算量小，FSDP 的分片管理和额外通信占比高，DDP 简单直接，吞吐通常更好。大模型时，完整复制参数、梯度和 optimizer state 会把显存打满，FSDP 的分片能把模型从 OOM 变成可训练。

这就是 memory pressure 矩阵想说明的事情：优化不是绝对的，要看模型处在哪个规模区间。

### Q127. 面试官：怎么判断性能优化有没有污染 benchmark？

答：看实验协议是否变化。比如是否开启了 profiler、是否换了环境、GPU 是否被占用、warmup 和 measured steps 是否一致、repeat 是否独立初始化、provenance 是否完整。

如果这些没固定，优化结果就不可信。训练性能数字首先要回答“是否有效”，然后才讨论“是否更快”。

### Q128. 面试官：step time 下降但显存上升，这算好优化吗？

答：不一定。要看目标。如果目标是小模型吞吐，显存有余量，那可能可以接受；如果目标是大模型训练或稳定运行，显存上升可能让 job 更容易 OOM。

训练 Infra 里优化通常是多目标权衡：吞吐、显存、稳定性、恢复能力和复杂度。不能只看一个指标赢了。

### Q129. 面试官：如何优化 checkpoint 对训练吞吐的影响？

答：先测 checkpoint 保存耗时和频率，确认它占 step time 的比例。优化方向包括降低保存频率、异步或分层保存、只保留必要状态、优化存储带宽、以及 retention 策略。

但 checkpoint 不能为了快牺牲正确性。READY、metadata、RNG 和 scheduler 这类状态不能随便省掉。否则吞吐提高了，恢复却不可信。

### Q130. 面试官：MoE 训练性能怎么优化？

答：MoE 的性能瓶颈通常包括 router 负载不均、token dispatch all-to-all、expert 计算不均、capacity overflow 和 combine 开销。优化要同时看通信和负载均衡。

MiniTrainBench 没实现完整 MoE layer，但它测了 all-to-all equal/uneven split 和 toy routing。面试里我会把这说成 MoE 通信语义证据，不会说成完整 MoE 训练优化。

### Q131. 面试官：Tensor Parallel 的性能瓶颈在哪里？

答：TP 的瓶颈通常在层内 collective，比如 column/row parallel 的 gather 或 reduce，以及序列并行引入的额外通信。它能降低单卡参数和激活压力，但也会增加同步点。

本项目的 toy TP 是 correctness demo，不是性能 benchmark。面试里我会先讲切分语义，再讲真实 Megatron 中还需要 pipeline schedule、distributed optimizer 和 sharded checkpoint。

### Q132. 面试官：你怎么把 profiler 结果转成优化动作？

答：我会先把 profiler 结果归类：data 慢就优化输入；CPU op 多就减少 Python/同步开销；CUDA matmul 慢就看模型和 kernel；NCCL 占比高就看 collective 类型、调用次数和 rank spread；optimizer 慢就看 optimizer 实现或 ZeRO/FSDP 配置。

然后只改一个变量做实验，并用 repeat 和 provenance 验证。Profiler 的价值不是生成报告，而是指导下一步最小改动。

## 十二、面试答题模板

这一节不是题库，是临场组织语言的模板。

### Q133. 面试官：遇到不会的问题，你怎么答才不慌？

答：我会先承认边界，然后把自己知道的相关链路讲清楚。比如推理服务我没有完整落地，就说“这个项目主线是训练 Infra，但我可以从 KV cache、batching、latency 和 SLA 角度分析推理系统”。

不要把没做过的说成做过。Infra 面试里，诚实说明边界反而更可信。

### Q134. 面试官：你回答系统题时的通用结构是什么？

答：我会按四步说：先讲目标，再讲关键组件，再讲数据/状态流，最后讲故障和观测。比如训练平台题，我会先说目标是稳定高效运行训练 job，再讲调度、launcher、checkpoint、monitoring，最后讲 preemption、NCCL、OOM 和 provenance。

这样回答不容易散，也能让面试官继续追问具体模块。

### Q135. 面试官：你回答故障题时的通用结构是什么？

答：我会按“现象 -> 定位 -> 根因 -> 修复 -> 验证 -> 边界”回答。先描述看到什么，再说收集什么证据，然后给可能根因，接着说怎么修，最后说明怎么证明修好了，以及哪些还没证明。

这个结构很适合训练 Infra，因为训练问题经常跨 launcher、container、runtime、collective、checkpoint 和 report 多层。

### Q136. 面试官：你回答性能题时的通用结构是什么？

答：我会先问目标指标，是吞吐、显存、延迟还是稳定性。然后拆 step breakdown，定位瓶颈，再提出一个最小改动，最后用 repeat、profiler 和 provenance 验证。

性能题最忌讳直接给玄学参数。一个好的回答应该体现方法论：先量化，再定位，再实验，再确认副作用。

## 十三、字节 SeedInfra 训练实习专项背诵面经

这一节专门按 SeedInfra 训练实习准备。公开信息里，Seed Infrastructures 方向覆盖分布式训练、强化学习框架、高性能推理和异构硬件编译器；Seed Careers 里有 LLM Infra training 和 Inference Optimization 岗位；ByteRobust 论文公开讨论了 CUDA error、NaN、job hang、故障诊断、故障恢复和 ETTR。这里按公开信息做面试准备，不写内部面经，也不把 MiniTrainBench 没实现的能力说成已经实现。

参考链接：[Seed Infrastructures](https://seed.bytedance.com/direction/infrastructures)、[Seed Careers](https://seed.bytedance.com/en/career)、[Seed Campus Recruitment](https://seed.bytedance.com/en/seedearlycareer)、[ByteRobust](https://arxiv.org/abs/2509.16293)。

### 13.1 岗位匹配与项目映射

### Q137. 面试官：你理解 SeedInfra 训练实习最看重什么？

答：我理解它不是单纯找会写 PyTorch 训练脚本的人，而是看候选人能不能理解大模型训练背后的系统问题：分布式并行、通信、显存、checkpoint、故障恢复、性能分析、自动化工具和平台稳定性。训练 Infra 的价值不是让一个 demo 跑起来，而是让长时间、大规模、多角色的训练持续有效地跑。

所以我准备这个岗位时，会把 MiniTrainBench 讲成训练 runtime contract 的证据：它虽然不是生产平台，但覆盖了 DDP/FSDP/ZeRO、checkpoint/resume、profiler、NCCL/MoE 通信、故障 smoke 和 provenance。这些点和 SeedInfra 公开方向里的 distributed training、RL framework、inference optimization、heterogeneous hardware compilation 是同一类系统能力。

### Q138. 面试官：怎么把 MiniTrainBench 讲成适合 SeedInfra 的项目？

答：我会先说它对齐的是训练系统的核心链路，而不是模型算法。项目用 synthetic data 固定数据变量，用最小 GPT-like 模型触发真实训练路径，再对比 DDP、FSDP 和 ZeRO 在不同 GPU 数、显存压力和通信模式下的行为。

更重要的是，它不只看 tokens/sec。我做了 checkpoint 原子发布、RNG 精确恢复、rank crash 后手工恢复、profiler 跨 rank 摘要、NCCL collective benchmark 和 MoE all-to-all demo。SeedInfra 面试里我会强调：这个项目证明的是我能把训练系统的状态、性能和故障边界拆开验证。

### Q139. 面试官：给你 30 秒，面向 SeedInfra 怎么介绍自己？

答：我主要准备的是大模型训练 Infra 方向，核心项目是 MiniTrainBench。它是一个小型分布式训练 runtime，用 PyTorch DDP/FSDP/DeepSpeed ZeRO 跑 GPT-like 训练，并围绕 checkpoint/resume、profiler、NCCL 通信、MoE all-to-all、显存压力和故障恢复做证据链。

我想投 SeedInfra，是因为我更关注训练系统怎么稳定、可复现、可诊断地运行，而不是只追模型指标。这个项目能说明我对分布式训练 runtime、性能定位和故障边界有比较系统的理解。

### Q140. 面试官：给你 90 秒，面向 SeedInfra 怎么介绍项目？

答：MiniTrainBench 的目标是做一个最小但可信的训练 runtime。它不依赖真实数据集，而是用 deterministic synthetic token 固定输入，重点观察训练系统本身：DDP/FSDP/ZeRO 的通信和显存取舍、gradient accumulation 的同步语义、checkpoint/resume 的状态完整性、profiler 的瓶颈定位，以及 MoE all-to-all 这种大模型训练里常见的通信模式。

它已经有 1/2/4/8 卡 benchmark、repeat=3 稳定性矩阵、memory pressure 矩阵、rank crash smoke 和 checkpoint exact verify。面试里我会补边界：它不是万卡生产平台，也没有实现 RLHF/GRPO，但它把训练 Infra 最基本的 contract 做成了可复现实验，这正是我想进入 SeedInfra 后继续放大的方向。

### Q141. 面试官：如果给你 3 分钟，你会展开哪三条主线？

答：第一条主线是分布式训练策略。我会讲 DDP 简单高吞吐、FSDP/ZeRO 通过分片降低显存、不同模型规模下策略选择不同，以及为什么小模型结论不能外推到大模型。第二条主线是状态和恢复。我会讲 checkpoint 保存 model、optimizer、scheduler、TrainState、RNG，为什么 READY/latest/fingerprint 能避免半成品恢复。

第三条主线是诊断和证据。我会讲 profiler 为什么单独跑、为什么 repeat 和 provenance 重要、为什么通信 benchmark 要单独拆出来。最后收束到岗位：SeedInfra 面对的是更大规模、更复杂角色和更高稳定性要求，而 MiniTrainBench 是我对这些问题做的缩小版验证。

### Q142. 面试官：为什么你不投泛后端，而是投训练 Infra？

答：因为我更感兴趣的是训练系统里算法、硬件和分布式工程交界的部分。泛后端也重视可靠性和性能，但训练 Infra 多了 GPU、NCCL、并行策略、checkpoint、显存峰值、数值稳定性这些特殊约束。

我的项目选择也能说明这个偏好。我没有去做一个普通 Web 服务，而是围绕训练 step、rank、collective、profiler 和恢复语义做工具。SeedInfra 对我吸引力最大的地方，就是这些问题既有工程复杂度，也有很强的系统研究味道。

### Q143. 面试官：你没有万卡训练经验，怎么证明能做这个岗位？

答：我会承认没有万卡经验，但强调自己准备的是可迁移的系统能力。万卡训练当然有更复杂的调度、拓扑、故障率和自动化要求，但底层问题仍然是状态一致性、同步通信、显存控制、性能诊断和故障恢复。

MiniTrainBench 的价值是把这些问题缩小到单机多卡里可复现：我能解释为什么某个 rank 慢会拖全局，为什么 checkpoint 要有发布协议，为什么 profiler 结果不能直接当 benchmark，为什么 FSDP 小模型慢但大模型有价值。进入团队后，我需要补的是生产规模和平台经验，而不是从零理解训练系统。

### Q144. 面试官：SeedInfra 公开提到训练、RL、推理、编译器，你怎么对齐？

答：我会分层对齐。训练是我当前项目的主线，DDP/FSDP/ZeRO、checkpoint、profiler 和 NCCL 都有证据；RL/post-training 我会作为知识储备去讲 rollout、trainer、reward model、reference model 和权重同步，不伪装成项目实现；推理我会讲 prefill/decode、KV cache、batching、SLA，是系统能力迁移；编译器和 CUDA/Triton 我会讲自己知道性能优化需要 kernel 和硬件视角，但还需要补实战。

这样回答的重点是诚实分层。面试官能看到我既知道岗位全貌，也知道自己的项目边界，不会把“了解过”说成“落地过”。

### Q145. 面试官：你的项目最能体现实习上手速度的地方是什么？

答：我觉得是把问题做成可复现工具的能力。比如 checkpoint 不是只写一个保存函数，而是设计 READY/latest、metadata、fingerprint、verify 和 crash smoke；profiler 不是只打开 PyTorch profiler，而是把 rank spread、NCCL op、step breakdown 和 provenance 放到报告里。

实习真正需要的是快速定位问题、快速做小实验、快速把结论写成别人能复查的证据。MiniTrainBench 的结构说明我有这个习惯：不只是修一个 bug，而是把 bug 背后的系统 contract 补上。

### Q146. 面试官：如果你入职 SeedInfra，第一个月你会怎么切入？

答：我会先从团队现有工具和线上问题切入，而不是一上来重写系统。第一周看训练 job 生命周期、launcher、checkpoint、监控、profiling 和常见故障工单；第二周找一个低风险问题做复现，比如某类 NCCL timeout、OOM、checkpoint 慢或者性能抖动；第三到四周把定位过程固化成脚本、dashboard、preflight check 或文档。

我能贡献的不是“马上设计万卡平台”，而是用扎实的训练 Infra 方法论提高排障效率。MiniTrainBench 里已经练过这种路径：先复现，再归因，再加验证，再沉淀工具。

### 13.2 大规模训练稳定性和 ByteRobust 风格追问

### Q147. 面试官：ETTR 是什么，为什么训练 Infra 要关心它？

答：ETTR 可以理解成 Effective Training Time Ratio，也就是有效训练时间占总时间的比例。大规模训练里，光看 job 是否最终成功不够，因为一个任务可能反复 hang、重启、恢复，最后虽然成功，但浪费了大量 GPU 时间。

所以 SeedInfra 这种方向会关心“训练是不是持续有效推进”。MiniTrainBench 还没有生产 ETTR 体系，但它已经有 related building blocks：repeat 统计、rank crash smoke、checkpoint READY、resume exact verify 和 provenance。面试里我会说自己理解 ETTR 的价值，但不会说项目已经实现 ByteRobust 级别的系统。

### Q148. 面试官：MTBF、MTTR、ETTR 怎么区分？

答：MTBF 更关注两次故障之间平均能跑多久，MTTR 更关注故障后平均多久恢复，ETTR 更关注整体训练时间里有多少比例是真正在推进训练。对大规模训练来说，MTBF 往往会随着 GPU 数上升而变差，所以只追求不出故障不现实。

更实际的目标是快速发现、快速隔离、快速恢复，让 ETTR 高。这个思路对应到项目里，就是失败时不能污染 READY checkpoint，恢复后要能 verify，性能实验要能区分有效结果和被污染结果。

### Q149. 面试官：大规模训练常见故障应该怎么分类？

答：我会按层分类。硬件层包括 GPU error、ECC、机器掉线、网络抖动；通信层包括 NCCL timeout、rank mismatch、collective shape 不一致；训练层包括 NaN、OOM、loss 发散；平台层包括调度失败、镜像版本不一致、存储权限和 checkpoint 损坏；观测层包括日志缺失、指标缺字段和 trace 无法关联。

这样分类的好处是排障不会乱。MiniTrainBench 目前覆盖的是训练层、通信层和部分平台语义，比如 NaN fail-fast、NCCL doctor、rank crash、checkpoint verify；真实生产还要补硬件健康、自动隔离和调度器联动。

### Q150. 面试官：训练 hang 住了，你会怎么按大规模思路排查？

答：我先判断 hang 在哪里：启动、data、forward、backward、optimizer、checkpoint 还是 collective。然后看是不是所有 rank 都停住，还是某些 rank 退出、某些 rank 等待。如果停在 collective，重点看 rank 是否都进入同一个 collective、tensor shape 是否一致、NCCL 网络是否正常。

大规模场景还要做 stack aggregation 和 rank 分组观察，因为单看一个 rank 的日志很容易误判。MiniTrainBench 里我能用 rank 日志、doctor、comm benchmark 和 fault smoke 做小规模复现；生产系统则需要自动采集 stack、NCCL 日志和节点健康。

### Q151. 面试官：NaN 问题为什么不能只在单卡 debug？

答：单卡能定位数值稳定性的一部分问题，比如 lr 太大、loss scale 不合适、某个 op 溢出。但分布式 NaN 还可能来自某个 rank 数据异常、梯度同步后污染全局、混合精度 reduce、FSDP/ZeRO state 不一致，或者恢复后 scheduler/RNG 不一致。

所以我会先做 all-rank finite check，确认 NaN 是本地出现还是同步后扩散。MiniTrainBench 里的 NaN 注入和 fail-fast 验证说明我理解这个点：分布式训练里不能让一个 rank 的坏状态悄悄写进 checkpoint。

### Q152. 面试官：CUDA error 或 GPU 节点故障怎么处理？

答：我会先区分是应用层 CUDA error，还是硬件/驱动/节点问题。应用层可能是 illegal memory access、shape 问题、kernel 参数错误；硬件层可能是 GPU unhealthy、ECC、driver reset 或节点网络异常。前者要回到代码和输入复现，后者要隔离节点并触发 job 恢复。

训练 Infra 的关键不是“看到 CUDA error 就重启”。如果根因是坏卡，简单重启可能反复失败；如果根因是代码 bug，换机器也没用。生产系统要有故障 demarcation，MiniTrainBench 目前只能模拟 rank crash，不承担硬件级自动隔离。

### Q153. 面试官：为什么同步训练里一个慢 rank 会拖住全局？

答：因为同步训练在 backward、optimizer 或 pipeline stage 上有同步点。比如 DDP 的 all-reduce 要等所有 rank 的梯度，FSDP 的 reduce-scatter/all-gather 也需要相关 rank 参与；一个 rank 慢了，其他 rank 即使算完也要等。

这就是 straggler 对 LLM 训练影响大的原因。面试里我会把它和 MiniTrainBench 的 rank spread、collective time、step max 聚合联系起来：分布式 step time 应该看慢的 rank，而不是平均 rank。

### Q154. 面试官：straggler 一定是硬件坏了吗？

答：不一定。straggler 可能来自硬件，也可能来自网络拥塞、进程调度、数据路径、checkpoint I/O、温度降频、其他进程干扰，甚至是某个 rank 的 workload 不均衡。MoE 里 router 负载不均也会制造 straggler。

所以我不会一上来就说换机器。我会先用 what-if 思路问：如果去掉这个慢 rank 或慢时间段，整体训练会改善多少；再看慢是否有时间/空间规律。MiniTrainBench 的 uneven all-to-all 和 rank spread 是这个思路的小型版本。

### Q155. 面试官：你怎么理解 what-if analysis 在 straggler 诊断里的价值？

答：what-if analysis 的价值是把“感觉某个 rank 慢”变成“这个慢 rank 对整体训练造成了多少损失”。它不是只找最慢点，而是模拟如果没有这个慢点，job 的 step time 或有效训练时间会怎样。

在面试里我会说，自己项目还没实现完整 what-if 分析，但已经在收集能支持它的数据：每 rank step time、collective time、profiler breakdown、repeat 和 provenance。下一步可以把这些数据用于自动判断 straggler 是否值得调度迁移或故障隔离。

### Q156. 面试官：大规模训练的 checkpoint 快速恢复系统怎么设计？

答：我会把它拆成四件事：保存频率、发布协议、发现逻辑和恢复校验。保存频率要平衡训练吞吐和故障损失；发布协议要保证半成品不可见；发现逻辑要快速找到最新可用点；恢复校验要确认 model、optimizer、scheduler、RNG 和并行拓扑匹配。

MiniTrainBench 的 READY/latest/metadata/verify 是最小实现。大规模生产还要补对象存储、多副本、跨节点带宽、checkpoint eviction、防过度清理和自动拉起。面试里我会明确这是扩展方向，不说项目已经做了。

### Q157. 面试官：故障自动诊断系统应该输出什么？

答：它不应该只输出“失败了”，而应该输出失败类别、可能层级、受影响 rank/节点、最后活跃阶段、可恢复性和建议动作。比如是 NCCL timeout，就要给 rank 集合、网卡/节点信息、最后 collective；是 NaN，就要给首个出现 NaN 的 rank、step、loss/grad norm 和最近 checkpoint。

输出要能驱动下一步：自动重试、隔离节点、回滚 checkpoint、拒绝恢复或交给人工。MiniTrainBench 目前的 fault smoke 和 report 是人工可读证据，还不是自动诊断系统，但方向一致。

### Q158. 面试官：恢复之后怎么证明没有污染训练？

答：先看恢复点是不是 READY checkpoint，再看 config fingerprint、world size、strategy、precision 是否匹配。然后比较 model、optimizer、scheduler、TrainState 和 RNG digest，最后跑连续训练和中断恢复的对照，看 loss 或 checkpoint 是否 exact match。

生产里不一定每次都能做 exact match，因为真实数据、异步组件和非确定性 kernel 会增加难度。但至少要能证明恢复没有从半成品开始，没有丢 optimizer/scheduler 状态，没有把失败 step 写进 latest。MiniTrainBench 的 exact verify 就是这个答案的项目证据。

### Q159. 面试官：什么时候应该 fail fast，什么时候应该容错继续？

答：如果错误会污染训练状态，比如 NaN、梯度 Inf、checkpoint 半成品、rank 不一致，我倾向 fail fast，因为继续训练可能浪费更多 GPU 并产出不可用模型。如果是可隔离的节点故障、短暂网络抖动或非关键观测失败，可以考虑容错或降级。

关键判断标准是状态是否可信、恢复成本是否低、继续运行是否会扩大损害。MiniTrainBench 在 NaN 和 half checkpoint 上选择保守策略：宁可失败，也不推进 READY 和 latest。

### Q160. 面试官：训练稳定性 dashboard 应该放哪些核心图？

答：我会放五类：有效训练进度，比如 global_step、tokens_seen、ETTR；性能趋势，比如 step time、tokens/sec、rank spread；资源健康，比如 GPU 利用率、显存、网络、节点状态；通信诊断，比如 NCCL op 时间和 timeout；状态持久化，比如 checkpoint 保存耗时、latest age、恢复成功率。

dashboard 的目标不是好看，而是减少定位时间。看到 step time 抖动时，应该能继续点到 rank、阶段、collective 和节点；看到失败时，应该知道是可恢复、需隔离还是要人工介入。

### 13.3 分布式并行、通信和显存深挖

### Q161. 面试官：DP、TP、PP、SP、EP 分别解决什么问题？

答：DP 数据并行复制模型、切数据，主要扩吞吐；TP 张量并行切单层矩阵，解决单层参数或计算太大；PP 流水并行切模型层，解决模型层数太多放不下；SP 序列并行切 activation，缓解长序列显存；EP 专家并行让不同 rank 持有不同 expert，常见于 MoE。

真实大模型训练通常不是单选，而是混合。比如 TP/PP/DP 组合后还可能加 SP 和 EP。MiniTrainBench 做的是 DDP/FSDP/ZeRO 主线，加 toy TP/SP 和 MoE communication demo，用来证明我理解这些并行语义，但没有实现完整 Megatron 训练。

### Q162. 面试官：Megatron-style 并行为什么复杂？

答：复杂在它不是只把模型切开，而是要同时管理张量切分、pipeline schedule、micro-batch、activation、通信 overlap、optimizer state、checkpoint shard 和随机数一致性。一个维度切错，可能 forward 能跑，backward 或 checkpoint 就不对。

我项目里的 Megatron 相关是 external compatibility smoke 和 toy correctness，不是完整框架。面试里我会重点讲语义：Column/Row Parallel Linear 为什么要配合、PP 为什么有 bubble、SP 为什么省 activation，而不夸大成“我实现了 Megatron”。

### Q163. 面试官：TP 里的主要 collective 在哪里？

答：以 MLP 为例，Column Parallel Linear 通常把输出维度切到不同 rank，每个 rank 计算部分 hidden；Row Parallel Linear 再把输入维度切开，最后需要 reduce 得到完整输出。attention 里也会有类似的 all-gather、reduce-scatter 或 all-reduce 语义，取决于具体切分。

TP 的问题是通信发生在层内，频率高、路径短，对 overlap 和 kernel 调度很敏感。MiniTrainBench 的 toy TP check 验证的是 forward/backward correctness，不能代表真实 TP 性能。

### Q164. 面试官：Pipeline Parallel 的 bubble 是什么？

答：PP 把不同层放到不同 stage，micro-batch 像流水线一样经过这些 stage。bubble 是流水线填充和排空时，有些 stage 没活干的空闲时间。micro-batch 越少、stage 越多，bubble 占比越明显。

优化 PP 要看 micro-batch 数、schedule、stage balance 和通信开销。MiniTrainBench 没实现 PP，所以面试里只能作为知识储备来讲，并把边界说清楚。

### Q165. 面试官：Sequence Parallel 节省的是什么显存？

答：SP 主要节省和 sequence length 相关的 activation 显存，尤其在长上下文训练里有价值。它通常和 TP 搭配，把原本每个 TP rank 都保留的某些 activation 沿 sequence 维度切开。

但 SP 不是免费午餐，它会引入额外通信和实现复杂度。MiniTrainBench 有 toy sequence parallel correctness demo，可以证明我理解切分和还原语义，但没有把它放进主训练 runtime。

### Q166. 面试官：MoE 为什么核心通信是 all-to-all？

答：MoE 里每个 token 会被 router 分到某些 expert，而 expert 分布在不同 rank 上。一个 rank 上的 token 可能要发给其他 rank 的 expert，其他 rank 的结果也要发回来 combine，这就是 token dispatch/combine 的 all-to-all 语义。

和 dense model 的 all-reduce 不同，MoE 通信很容易不均衡。MiniTrainBench 的 comm benchmark 专门测 equal 和 uneven split，就是为了把均衡和不均衡两种情况拆开讲。

### Q167. 面试官：MoE load balance 为什么会影响训练性能？

答：如果 router 把大量 token 分到少数 expert，这些 expert 所在 rank 就会成为 straggler，其他 rank 要等它完成 dispatch、expert compute 和 combine。这样即使平均 tokens/sec 看起来不错，尾部 rank 也会拖慢整个 step。

所以 MoE 优化不能只看总通信带宽，还要看每个 expert 的 token 数、overflow、capacity 和 rank spread。MiniTrainBench 的 toy routing 能展示 load imbalance 和 overflow 语义，但不是完整 MoE 层性能优化。

### Q168. 面试官：capacity 和 overflow 在 MoE 里是什么意思？

答：capacity 是每个 expert 在一个 batch 中最多接收多少 token，通常用于限制最坏情况显存和计算量。overflow 是 token 分配超过 expert capacity 后被丢弃、转发到备用 expert，或走其他处理策略。

系统角度看，capacity 是稳定性和质量之间的权衡。capacity 太小可能影响训练质量，太大可能让显存和 step time 抖动变大。面试里要把它讲成 runtime tradeoff，而不是只讲 router 算法。

### Q169. 面试官：MoE 的通信计算 overlap 怎么做？

答：基本思路是不要等所有 token dispatch 完再开始 expert compute，而是把 token 分块，让部分通信和部分 expert 计算重叠。更细的做法还会按依赖关系调度，把小块通信、expert compute 和 combine 排得更紧。

但 overlap 的难点是不能牺牲正确性和计算效率。切得太细会增加调度开销，切得太粗又隐藏不了通信。MiniTrainBench 当前只能测 all-to-all 形态，不能宣称实现了 fine-grained MoE overlap。

### Q170. 面试官：FSDP 和 ZeRO-3 为什么经常放在一起比较？

答：因为它们都试图分片参数、梯度和 optimizer state，从而降低每张卡的显存压力。区别在于工程栈、状态管理、checkpoint 语义、配置方式和和 PyTorch runtime 的集成方式不同。

我项目里把 FSDP 接进主 Trainer，把 DeepSpeed ZeRO 做成独立 adapter，就是为了避免两套 engine 生命周期混在一起。面试里这个设计点很适合讲工程边界：可比 benchmark 可以做，但 checkpoint 语义不能乱合并。

### Q171. 面试官：activation checkpointing 和 FSDP 一起用有什么坑？

答：activation checkpointing 通过重算节省 activation 显存，FSDP 通过分片节省参数、梯度和 optimizer state。两者叠加可以降低显存峰值，但会增加重算和通信调度复杂度。

坑在于峰值可能转移，不一定消失。比如 FSDP all-gather 参数时仍可能产生通信峰值，activation 重算也会改变 step breakdown。MiniTrainBench 里把 BF16、activation checkpointing 和 accumulation 组合跑 smoke，是为了验证这些开关不会破坏基本训练语义。

### Q172. 面试官：gradient accumulation 为什么在 DDP 和 FSDP 下策略不同？

答：DDP 下常用 `no_sync()`，让前几个 micro-batch 不 all-reduce，只在最后同步，减少通信次数。FSDP 下如果也这么做，可能保留未分片梯度，显存峰值会上升，所以默认策略要更谨慎。

这就是为什么 MiniTrainBench 有 `gradient-sync-mode auto/every/last` 的价值。它不是简单开关，而是在训练吞吐、通信次数和显存峰值之间做策略选择。

### Q173. 面试官：混合并行下 checkpoint 为什么更难？

答：因为 checkpoint 不再只是一个完整 state dict。TP/PP/EP/FSDP/ZeRO 都可能改变参数、optimizer state 和 RNG 的分布方式。恢复时不仅要知道 step，还要知道并行拓扑、shard layout、rank mapping 和版本信息。

如果这些元数据缺失，checkpoint 目录存在也不代表可恢复。MiniTrainBench 的 fingerprint、metadata 和 verify 是小规模版本；真实混合并行还需要更强的 manifest 和 resharding 逻辑。

### Q174. 面试官：训练吞吐低时，怎么按并行维度排查？

答：先看瓶颈在哪个阶段。如果 data 慢，先别调并行；如果 forward/backward 慢，看 kernel、activation checkpointing 和 TP 切分；如果 communication 慢，看 DP all-reduce、FSDP all-gather/reduce-scatter、MoE all-to-all 或 PP stage 通信；如果 optimizer 慢，看 ZeRO/FSDP state 和 optimizer 实现。

然后看 rank spread 和 overlap。并行训练里吞吐低经常不是单个 op 慢，而是某个同步点让所有 rank 等待。MiniTrainBench 的 profiler 和 comm benchmark 正好能支撑这种排查口径。

### 13.4 RL / Post-Training 系统面经

### Q175. 面试官：pretraining Infra 和 post-training Infra 最大区别是什么？

答：pretraining 更像长时间、稳定数据流的同步训练，核心关注吞吐、显存、checkpoint、故障恢复和数据顺序。post-training，尤其 RLHF/GRPO，会把训练、推理 rollout、reward、reference model、数据过滤和策略更新交织在一起。

所以 post-training Infra 更像多角色系统：一部分角色在训练，一部分角色在生成，一部分角色在打分或管理样本。MiniTrainBench 主线是 pretraining runtime，这部分我会作为岗位知识储备讲，不说成项目已经实现。

### Q176. 面试官：PPO/GRPO 训练系统里通常有哪些角色？

答：常见角色包括 actor 或 policy model、rollout worker、trainer、reward model、reference model、数据队列和调度控制器。actor 负责生成或更新策略，rollout 负责采样，reward model 给反馈，reference model 用于 KL 约束或对照，trainer 负责梯度更新。

系统难点是这些角色的资源需求不同。rollout 更像推理，关注生成吞吐和 KV cache；trainer 更像训练，关注反向传播和 checkpoint；reward model 可能成为服务瓶颈。面试里要把它讲成系统编排问题，而不是只背 PPO 公式。

### Q177. 面试官：rollout、trainer、reward model、reference model 分别卡在哪里？

答：rollout 常卡在生成吞吐、decode latency、KV cache 和请求调度；trainer 卡在反向传播、分布式同步、显存和 optimizer；reward model 卡在打分吞吐、batching 和数据排队；reference model 卡在额外推理成本和权重版本一致性。

如果让我设计观测，我会按角色采不同指标，而不是只看一个 global tokens/sec。RL 系统的问题经常是局部角色背压传导到全局，例如 reward 慢导致 rollout 队列堆积，最后 trainer 等数据。

### Q178. 面试官：为什么 RL 训练比 pretraining 更难做容错？

答：因为 pretraining 的状态主要是 model、optimizer、scheduler、RNG 和数据进度；RL 还多了 rollout buffer、样本版本、reward 结果、reference policy、actor 权重同步和各角色队列。失败时你要判断哪些状态可以保留，哪些必须丢弃。

比如 trainer 重启后，旧 rollout 是否还能用，取决于 policy version 和算法容忍度；rollout worker 重启后，是否需要重放也取决于数据一致性。MiniTrainBench 的 checkpoint/resume 思路能迁移，但 RL 需要 role-aware recovery。

### Q179. 面试官：RL 系统里的 weight sync 为什么重要？

答：rollout worker 生成样本时用的是某个版本的 policy，trainer 更新后需要把新权重同步给 rollout 或 inference worker。如果同步慢，rollout 用旧策略生成太多样本，训练就会变 stale；如果同步方式粗糙，又会浪费大量网络和显存。

系统设计上要明确权重版本、同步频率、传输方式和一致性要求。这个点和 MiniTrainBench 的 checkpoint metadata 类似：状态不只是 bytes，还要有版本语义。

### Q180. 面试官：RL 里 trainer 失败和 rollout 失败应该一样处理吗？

答：不应该。trainer 失败通常影响模型更新状态，需要从 checkpoint 或最近一致点恢复；rollout 失败更多影响样本生成，可以局部替换或重启 worker。reward model 失败又可能导致样本打分阻塞。

这就是 role-aware fault tolerance 的思路：不要所有故障都全局重启。MiniTrainBench 目前只有训练 rank crash，不能覆盖 RL 多角色恢复，但它保存 READY checkpoint、不污染 latest 的原则可以迁移。

### Q181. 面试官：RL 训练里的数据 provenance 要记录什么？

答：至少要记录 prompt 来源、policy version、rollout 参数、reward model 版本、打分结果、过滤规则、采样时间和训练 step。否则后面 loss 异常或 reward 异常时，很难判断是模型问题、数据问题还是 reward 版本变化。

MiniTrainBench 记录 provenance 是为了让 benchmark 可复查；RL 里 provenance 更重要，因为数据是在线生成的，不是固定静态语料。面试里可以把这作为从训练 runtime 到 post-training runtime 的自然扩展。

### Q182. 面试官：reward model 成为瓶颈时怎么优化？

答：先确认瓶颈是模型推理慢、batching 不够、队列调度差，还是数据预处理/后处理慢。优化方向包括动态 batching、并行 reward worker、缓存、模型量化、异步打分和背压控制。

但 reward 优化不能只看吞吐，还要看一致性和质量。换 reward model 版本、改 batch 策略或使用缓存，都可能改变训练信号。面试里我会把它讲成“性能和训练语义一起验证”。

### Q183. 面试官：MiniTrainBench 的经验怎么迁移到 RL post-training？

答：最直接迁移的是四件事：状态完整性、性能分解、故障边界和证据记录。状态完整性对应 checkpoint 和 weight version；性能分解对应 trainer/rollout/reward 的 breakdown；故障边界对应 role-aware recovery；证据记录对应 provenance。

我会强调迁移不是复用代码，而是复用工程方法。MiniTrainBench 还没有 rollout worker 和 reward model，但它训练我用最小实验验证系统 contract，这对 RL Infra 很有用。

### Q184. 面试官：如果被问“你做过 RLHF/GRPO 系统吗”，你怎么答？

答：我会直接说没有完整实现过 RLHF/GRPO 系统，当前项目主线是 pretraining runtime。但我会补充自己已经准备了 RL 系统视角：知道 rollout、trainer、reward、reference、weight sync、role-aware fault tolerance 和数据 provenance 是关键。

然后我会把话题落回可证明能力：我做过 checkpoint/resume、profiler、分布式同步、MoE 通信和故障 smoke。它们不是 RL 系统本身，但都是进入 RL training infra 后能迁移的基础能力。

### 13.5 平台调度、Ray/KubeRay、数据与自动化

### Q185. 面试官：Ray 在训练 Infra 里通常解决什么问题？

答：Ray 更偏分布式任务编排和 actor 管理，适合组织多角色 workload，比如训练、rollout、reward、数据处理和评估。它不是替代 PyTorch/NCCL 的底层通信，而是在更高层管理任务、资源和生命周期。

面试里我会避免说“Ray 能解决所有分布式训练问题”。更准确的说法是：GPU 训练内部仍然依赖 torchrun、NCCL、FSDP/Megatron 等机制；Ray 可以帮助编排异构角色、弹性任务和 pipeline。

### Q186. 面试官：K8s + GPU 训练平台要考虑什么？

答：首先是资源语义：GPU 型号、数量、显存、拓扑、IB/RDMA、节点健康和镜像环境。其次是作业语义：gang scheduling、rank 注入、rendezvous、日志、checkpoint 存储、失败重试和队列公平性。

训练 job 和普通 stateless 服务不一样。它需要多个 pod 同时 ready，rank/world size 不能乱，某个 worker 失败可能导致全局退出。MiniTrainBench 的 doctor 和 launcher 模板只能覆盖小部分，生产平台还要和调度器深度结合。

### Q187. 面试官：为什么训练任务常需要 gang scheduling？

答：因为分布式训练通常要求一组 GPU 或节点同时启动。如果只拿到部分资源，rank 不完整，job 不是慢一点，而是根本无法正确进入 collective。特别是 TP/PP/DP 混合并行时，拓扑不完整会直接破坏并行组。

所以调度器需要理解训练 job 的整体资源需求，而不是把每个 worker 当独立任务。面试里可以说，gang scheduling 是把分布式训练的同步语义传递给平台调度。

### Q188. 面试官：elastic training 是不是所有问题的答案？

答：不是。Elastic 能在某些 DP 场景下处理 worker 增减，但大模型混合并行不一定容易 elastic。TP/PP/EP 的拓扑、optimizer shard、checkpoint layout 和 global batch 都可能依赖固定 world size。

所以是否 elastic 要看训练策略和状态设计。MiniTrainBench 目前 checkpoint verify 要求相同 world size，这就是保守边界。面试里我会说，先保证固定拓扑恢复正确，再讨论 elastic resharding。

### Q189. 面试官：preflight 自动化检查应该覆盖什么？

答：我会覆盖镜像 digest、代码 commit、GPU 数和型号、driver/CUDA/NCCL 版本、网络接口、rendezvous 端口、磁盘空间、对象存储权限、GPU 空闲状态和基础 collective smoke。目标是在训练真正开始前暴露环境问题。

这个点很适合和 MiniTrainBench 的 `doctor` 联系起来。当前 doctor 是小型环境诊断，生产里可以扩展成提交前 gate：不满足条件就拒绝启动，避免把 GPU 时间浪费在明显失败的 job 上。

### Q190. 面试官：训练平台里的 quota、queue 和 fairness 怎么讲？

答：训练资源昂贵，平台必须控制谁能用多少 GPU、排队策略是什么、抢占规则是什么、优先级如何定义。fairness 不是简单平均分，因为不同任务可能有不同截止时间、规模和业务价值。

系统上要兼顾利用率和研发效率。低优任务可以被抢占，但必须有 checkpoint；高优任务可以插队，但不能长期饿死其他队列。训练 Infra 的调度策略必须和 checkpoint/resume 能力配套。

### Q191. 面试官：数据吞吐在训练 Infra 里为什么仍然重要？

答：虽然 MiniTrainBench 用 synthetic data 固定变量，但真实训练里数据读取、解码、shuffle、过滤和跨机读取都可能成为瓶颈。GPU 算得再快，如果 data loader 供不上，step time 还是会抖。

我在面试里会解释为什么项目不用真实数据：为了隔离训练 runtime 变量。但我不会否认数据管线重要。相反，如果接入生产平台，我会补 data time、I/O、cache hit、样本 provenance 和数据版本管理。

### Q192. 面试官：artifact 和 provenance 的 schema 应该怎么设计？

答：核心原则是让结果可复查。至少要记录代码 commit、镜像 digest、命令参数、硬件信息、框架版本、CUDA/NCCL 版本、seed、模型配置、训练策略、checkpoint path、性能指标和是否 performance_valid。

字段要区分必填和可选。比如 ZeRO adapter 和 DDP/FSDP 可能字段不同，报告层不能把缺字段当 0。MiniTrainBench 之前处理 ZeRO 报告 contract，就是这个问题的小型版本。

### Q193. 面试官：自动化 profiling 工具怎么设计？

答：我会让它分成触发、采集、聚合和建议四层。触发层决定什么时候开 profiler，采集层拿 trace、metrics、NCCL 日志和 rank 信息，聚合层做 step breakdown 和 rank spread，建议层把现象映射到可能动作，比如调 bucket、查数据、查网卡或看 optimizer。

同时 profiler 不能常态污染正式 benchmark。MiniTrainBench 把 profiler 命令和 benchmark 命令分开，就是为了区分“稳定数值”和“定位证据”。

### Q194. 面试官：怎么把一个 benchmark 项目升级成平台 service？

答：要补四层。第一是作业入口，把命令参数变成受控配置和 schema；第二是调度集成，让 GPU、节点和网络由平台分配；第三是状态服务，把 checkpoint、artifact、provenance 和报告集中管理；第四是诊断服务，把 failure 分类和 profiler 结果自动化。

MiniTrainBench 现在是 CLI 和脚本驱动，适合作为 runtime contract 和 evidence generator。升级成平台 service 时，不能丢掉它的确定性、repeat、READY 和 verify，这些反而应该成为平台的质量门槛。

### 13.6 高压追问和反问模板

### Q195. 面试官：这个项目规模很小，怎么证明价值？

答：我会说规模小是有意控制变量，不是能力上限。训练 Infra 项目如果一上来做大而全，很容易每个点都只停留在“能跑”。我选择小 runtime，是为了把 checkpoint、RNG、profiler、repeat、fault smoke 和通信语义做准。

然后我会主动承认边界：它不能证明我有万卡调度经验，但能证明我理解训练系统的关键 contract。对实习生来说，这比堆一个不可复现的大项目更有说服力。

### Q196. 面试官：你不会 CUDA/Triton 深优化，怎么做 SeedInfra？

答：我会承认目前 CUDA/Triton 不是最强项，但不会回避。SeedInfra 里确实有异构硬件编译和 kernel 优化方向，我当前更强的是训练 runtime、分布式状态、checkpoint、profiler 和故障诊断。

我会补一句：性能优化需要分层。不是所有问题都先写 kernel，很多训练问题先要定位是 data、NCCL、optimizer、显存峰值还是 kernel。本项目能证明我会用 profiler 和 benchmark 找瓶颈；如果瓶颈落到 kernel，我会继续补 Triton/CUDA。

### Q197. 面试官：你没做过生产系统，怎么说服我？

答：我会说我没有直接负责过生产万卡平台，但我在项目里刻意用了 production contract 的思路：半成品 checkpoint 不可见，恢复必须校验，性能结果必须有 provenance，正式 benchmark 和 profiler 分离，失败不能污染 latest。

这些原则和生产系统是一致的。差距主要在规模、自动化和平台联动，比如节点隔离、队列调度、对象存储、多租户和 SLO。我不会假装做过，但我能清楚说出要补什么。

### Q198. 面试官：如果只能再补一个能力，你会补什么？

答：如果目标是 SeedInfra 训练实习，我会优先补多机和自动诊断。多机能把 rendezvous、NCCL_SOCKET_IFNAME、网络拓扑、共享存储和 rank 分配问题暴露出来；自动诊断能把现有 doctor、profiler、fault smoke 和 report 串成更像平台的工具。

我不会优先补花哨模型结构。因为这个项目的定位是训练 Infra，继续增强应该围绕稳定性、诊断和平台化，而不是把模型做得更复杂。

### Q199. 面试最后你会怎么反问 SeedInfra 团队？

答：我会问四类问题。第一，团队当前最痛的是训练稳定性、吞吐、显存、调度还是 post-training 编排。第二，线上主要并行栈是 FSDP、Megatron、DeepSpeed、Ray 还是自研。第三，故障诊断是偏人工分析，还是已经有自动 demarcation 和 recovery。第四，实习生通常会从 profiling、工具链、bugfix 还是平台模块切入。

这些反问能体现我真的理解岗位，而不是泛泛问“团队氛围怎么样”。同时也能帮助我判断入职后该补哪块能力。

### Q200. 面试官：最后用一句话总结你为什么适合 SeedInfra 训练实习。

答：我适合这个方向，是因为我不是只会把训练脚本跑通，而是会把训练系统拆成状态、通信、显存、性能、故障和证据链来理解，并且已经用 MiniTrainBench 把这些点做成了可复现项目。

如果再压缩成一句话：我的优势是能用工程化方法把训练 Infra 的复杂问题变成可验证、可定位、可恢复的系统行为；我的边界是生产规模经验还需要在真实团队里继续补。

## 十四、基础知识点深挖面经

这一节补的是 SeedInfra 面试里容易被追问的基础知识。回答时不要只背定义，要尽量落到训练 Infra 场景：为什么会影响训练、怎么定位、怎么验证、项目里有没有证据。

### Q201. 面试官：Python 的 list、dict、set 在 Infra coding 里怎么选？

答：我会先按访问模式选。需要保持顺序和遍历，用 list；需要按 key 快速查找，用 dict；需要去重和成员判断，用 set。训练 Infra 里很常见，比如按 rank 聚合指标适合 dict，保存 step 时间序列适合 list，判断哪些 checkpoint 已经 READY 适合 set。

复杂度上，dict/set 平均查找是 O(1)，list 查找是 O(n)。面试 coding 时不要只说复杂度，还要说边界：key 是否可 hash、是否需要稳定顺序、数据量是否会撑爆内存。

### Q202. 面试官：Python GIL 对训练 Infra 有什么影响？

答：GIL 让同一个 Python 进程里的多个线程不能同时执行 Python bytecode，所以 CPU 计算密集型任务用线程不一定加速。但 I/O 密集型任务，比如读日志、拉文件、等网络，可以用线程隐藏等待。

PyTorch 训练里要区分 Python 层和 C++/CUDA 层。很多 tensor op 会释放 GIL 或直接异步交给 GPU，所以不能简单说“Python 有 GIL 所以 PyTorch 不能并行”。DataLoader 常用多进程 worker，也是为了绕开一部分 Python 预处理瓶颈。

### Q203. 面试官：什么时候用 iterator 或 generator？

答：当数据是流式的、不想一次全加载进内存时，我会用 iterator/generator。训练 Infra 里常见场景是读大日志、扫描结果 JSON、流式解析 profiler 事件，或者逐个产出 benchmark trial。

边界是 generator 只能消费一次，异常和资源释放要处理好。比如打开文件读日志时，要用 context manager 保证文件关闭；如果中途解析失败，要能报告行号和原始内容。

### Q204. 面试官：JSON schema 为什么重要？

答：因为训练平台很多组件靠 JSON 或结构化结果交互。benchmark runner、report、manifest、profiler summary、checkpoint metadata 如果字段语义不清，轻则报告崩溃，重则把缺失值当成 0，污染性能结论。

MiniTrainBench 里 ZeRO adapter 和 DDP/FSDP 结果字段不完全一样，所以报告层必须区分 required core fields 和 optional adapter fields。面试里我会强调：schema 是 Infra contract，不是文档装饰。

### Q205. 面试官：异常处理里什么时候应该捕获，什么时候应该抛出？

答：如果异常可以恢复，比如读取某个可选字段失败、某个非关键 artifact 缺失，可以捕获并给出明确 warning。如果异常会影响训练正确性，比如 checkpoint metadata 不匹配、RNG 缺失却声称 exact resume、loss NaN，就应该 fail fast。

训练 Infra 里最怕吞异常。表面上 job 继续跑，实际状态已经不可信。MiniTrainBench 在 half checkpoint、config mismatch、NaN 上都偏保守，就是这个原则。

### Q206. 面试官：文件原子写入为什么重要？

答：因为训练恢复逻辑经常通过目录或文件判断状态。如果写 checkpoint 或 manifest 时直接覆盖正式文件，中途失败就可能留下半成品，后续恢复进程会误读。

常见做法是先写临时路径，写完并校验后再 rename 或发布 READY 标记。MiniTrainBench 的 checkpoint 生命周期就是先临时目录，再 DCP 保存，最后 READY/latest。边界是对象存储不一定有 POSIX rename 语义，所以生产里可能需要 manifest 或 commit marker。

### Q207. 面试官：SIGTERM、SIGKILL、exit code 在训练平台里怎么理解？

答：SIGTERM 是可捕获的终止信号，进程有机会清理和保存状态；SIGKILL 不可捕获，进程会被直接杀死。exit code 是 launcher 或平台判断 job 成败的重要信号。

分布式训练里，一个 rank 被 SIGKILL 后，其他 rank 可能卡在 collective，所以 launcher 必须能传播失败。MiniTrainBench 的 rank crash smoke 验证的是 `SIGKILL -> torchrun 非零退出 -> 手工恢复 -> exact verify`，不是自动弹性恢复。

### Q208. 面试官：stdout、stderr 和日志 buffering 会带来什么坑？

答：训练任务常常依赖日志定位问题，但 stdout/stderr 可能被容器、launcher 或日志系统缓冲。进程 crash 时，最后几行日志可能还没 flush；多 rank 同时输出时，顺序也可能交错。

所以日志解析不能只假设“最后一行就是根因”。更好的做法是记录 rank、timestamp、step、stage 和 severity。MiniTrainBench 的 report 和 fault smoke 更偏结构化 JSON，也是为了减少纯文本日志的不确定性。

### Q209. 面试官：MASTER_ADDR、MASTER_PORT、rank、world size 是什么关系？

答：`MASTER_ADDR` 和 `MASTER_PORT` 是分布式 rendezvous 的入口，rank/world size 描述每个进程在全局训练中的身份和总数。所有 rank 必须对 world size 和 rendezvous 配置达成一致，否则很容易启动 hang。

多机训练里，这个问题经常先于模型问题出现。面试排障时我会先看环境变量、端口连通性、hostfile、rank 分配和网卡，再看模型代码。

### Q210. 面试官：Tensor 的 shape、stride、contiguous 为什么重要？

答：shape 决定逻辑维度，stride 决定内存里怎么走。一个 tensor 即使 shape 一样，也可能因为 transpose、slice 变成非 contiguous，某些 kernel 或 view 操作会失败或触发额外 copy。

训练 Infra 里这会影响性能和显存。比如 MoE token packing、TP shard、all-to-all buffer 都需要关心布局。coding 时如果要 reshape，我会先判断是否需要 `.contiguous()`，并说明这可能带来额外内存。

### Q211. 面试官：autograd graph、detach、no_grad 有什么区别？

答：autograd graph 记录 tensor 运算用于反向传播。`detach()` 是把某个 tensor 从当前 graph 里断开，`torch.no_grad()` 是在上下文里不记录梯度，常用于 eval、metric 或参数检查。

训练 Infra 中，错误使用 detach 可能让梯度断掉；忘记 no_grad 可能让评估或监控占用额外显存。面试里我会说，debug 梯度异常时不仅看 loss，也要检查哪些张量 `requires_grad`，以及参数的 grad 是否为 None。

### Q212. 面试官：为什么 CUDA 计时前后要 synchronize？

答：因为 CUDA kernel 通常是异步提交的，Python 代码返回不代表 GPU 已经执行完。如果直接用 `time.time()` 包一段 GPU 代码，测到的可能只是 kernel launch 时间，而不是实际计算时间。

正确做法是计时前后同步，或者用 CUDA event。MiniTrainBench 的 step time 会考虑 CUDA synchronize，并且分布式 step time 取跨 rank max，这样更接近同步训练真正等待的时间。

### Q213. 面试官：PyTorch DDP 的进程模型是什么？

答：DDP 通常是一卡一进程，每个 rank 持有完整模型副本，处理不同数据，backward 时通过 all-reduce 同步梯度。它不是一个进程里开多个 GPU 线程，而是多个独立进程协同。

这意味着日志、RNG、checkpoint shard、CUDA device、rank 环境都要按进程管理。MiniTrainBench 按 rank 保存 RNG state、按 rank 汇总 profiler，就是因为进程级状态不能只看 rank 0。

### Q214. 面试官：NCCL 和 Gloo 怎么选？

答：NCCL 主要面向 GPU collective，适合 CUDA tensor 的高性能通信；Gloo 更通用，CPU 场景和 CI smoke 里常用。训练大模型时，GPU 通信通常优先 NCCL。

边界是 CI 或本地没有 GPU 时，用 Gloo 做逻辑测试仍然有价值，但不能把 Gloo 结果当 NCCL 性能结论。MiniTrainBench 的 CPU CI 覆盖 Gloo correctness，GPU 脚本再验证 NCCL 性能和行为。

### Q215. 面试官：NCCL 环境变量你会先看哪些？

答：我会先看 `NCCL_SOCKET_IFNAME` 是否选对网卡，`NCCL_IB_DISABLE` 是否符合当前网络，`NCCL_DEBUG` 是否打开足够日志，`NCCL_ASYNC_ERROR_HANDLING` 或 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 是否帮助异步错误暴露。

但我不会靠背环境变量解决问题。正确顺序是先确认 rank/world size 和端口连通性，再用 doctor 或最小 collective benchmark 验证通信路径，最后回到训练 step。

### Q216. 面试官：BF16、FP16 和 GradScaler 你怎么讲？

答：BF16 指数范围接近 FP32，通常比 FP16 更不容易 overflow；FP16 精度更细但指数范围小，训练时常需要 GradScaler 防止梯度 underflow。选择哪种精度要看硬件支持、模型稳定性和性能目标。

checkpoint 里也要考虑精度状态。MiniTrainBench 主线是 BF16/FP32，所以没有保存 FP16 GradScaler state；如果扩展 FP16 mixed precision，就必须把 scaler 状态纳入 checkpoint，否则 resume 后数值路径可能漂移。

### Q217. 面试官：显存排查时 `allocated` 和 `reserved` 有什么区别？

答：`allocated` 更接近当前 tensor 实际占用，`reserved` 是 PyTorch caching allocator 向 CUDA 申请并保留的显存。reserved 高不一定等于内存泄漏，可能是缓存；但 allocated 持续增长就要警惕 graph 没释放、list 持有 tensor 或中间结果被保存。

训练 Infra 里要看峰值，而不是只看某一刻。OOM 可能发生在 FSDP all-gather、optimizer state 初始化、checkpoint 或 eval 阶段。验证时我会记录 max memory，并和 step/stage 对齐。

### Q218. 面试官：PyTorch Profiler 有哪些坑？

答：Profiler 会引入额外开销，改变调度和内存行为，所以不能默认打开后拿 tokens/sec 当正式性能结果。Profiler 更适合定位瓶颈，而 benchmark 更适合发布稳定数字。

另外，`key_averages()` 只给聚合统计，不保留完整时间线。要判断通信计算 overlap，需要看 trace 中 NCCL kernel 和 compute kernel 的时序。MiniTrainBench 把 profiler 和 benchmark 拆开，就是为了避免这两个用途混在一起。

### Q219. 面试官：CUDA/Triton 基础你怎么准备？

答：我会按性能定位需要来准备：线程块、warp、memory coalescing、shared memory、寄存器、occupancy、kernel launch overhead，以及 Triton 里的 program id、block size、mask 和内存访问模式。目标是能看懂为什么某个 kernel 慢，而不是一开始就声称自己能写复杂 fused kernel。

面试里如果被追 CUDA/Triton，我会诚实说当前项目没有实现自定义 kernel；我能用 profiler 定位到 kernel 层，并理解下一步要从访存、并行度和融合机会入手。这是岗位知识储备，不是项目已实现能力。

### Q220. 面试官：runtime 镜像和 devel 镜像有什么区别？

答：runtime 镜像通常包含运行 CUDA 程序需要的库，但不一定包含 nvcc、头文件和编译工具；devel 镜像才更适合编译 CUDA extension。很多 DeepSpeed、APEX、Transformer Engine 相关问题，其实是镜像能力不匹配。

MiniTrainBench 真实遇到过 runtime-only 镜像中 DeepSpeed import 探测 CUDA_HOME/nvcc 的问题。面试表达时我会说：不能把“PyTorch 能运行 CUDA”和“镜像能编译 CUDA 扩展”混为一谈。

## 十五、Coding 面经（训练 Infra 实战）

这一节按现场 coding 准备，主线是训练 Infra 实战题。回答格式建议固定成：题意、思路、边界、复杂度/验证。代码不需要背逐字实现，但要能现场写出核心逻辑。

### Q221. 面试官：写一个日志解析器，提取每个 step 的 loss、lr 和是否出现 NaN。

答：题意是从多行训练日志中抽结构化字段。思路是逐行读取，用正则或分隔规则提取 `step/loss/lr/rank`，不要一次读完整大文件。遇到 NaN/Inf 时记录首个异常 step 和 rank。

边界是日志可能乱序、字段缺失、多 rank 交错、某些行不是训练日志。验证时给正常行、缺字段行、NaN 行和多 rank 交错行。

```python
for line in stream:
    m = pattern.search(line)
    if not m:
        continue
    item = parse_match(m)
    by_step[item.step].append(item)
    if not isfinite(item.loss):
        first_bad = first_bad or item
```

复杂度是 O(n)，n 是日志行数。MiniTrainBench 更偏结构化 JSON，但真实平台仍经常需要这种日志兜底解析。

### Q222. 面试官：stdout 前面有 banner，后面才是 JSON，怎么稳健解析？

答：题意是命令输出不一定是纯 JSON。思路是不要假设 stdout 全部可 `json.loads()`，而是约定一个前缀，比如 `ENV_JSON=`，只解析带前缀的行，并保留原始 stdout 方便排查。

边界是多行 JSON、重复前缀、没有前缀、JSON 格式错误。验证时分别构造 banner、合法前缀、非法 JSON 和无前缀场景。

```python
prefix = "MINITRAINBENCH_ENVIRONMENT_JSON="
for line in stdout.splitlines():
    if line.startswith(prefix):
        return json.loads(line[len(prefix):])
raise ValueError("environment json not found")
```

这个题和项目里的 NGC banner 污染环境探针 JSON 很贴近。重点不是正则多复杂，而是输出协议要可恢复。

### Q223. 面试官：写一个 result JSON schema validator。

答：题意是检查 benchmark 结果能不能进入报告。思路是定义 required core fields，比如 `status`、`strategy`、`world_size`、`metrics`、`provenance`；adapter-specific 字段放 optional，不允许缺字段被默默当成 0。

边界是旧 JSON、ZeRO adapter 字段不同、性能无效但 smoke 成功、数值类型错误。验证时用 DDP/FSDP/ZeRO/legacy 四类样本。

```python
required = ["status", "strategy", "world_size", "metrics"]
for key in required:
    if key not in obj:
        errors.append(f"missing {key}")
if obj.get("performance_valid") is False:
    warnings.append("do not publish perf number")
```

复杂度是 O(k)，k 是字段数。面试里要强调 validator 保护的是性能结论可信度。

### Q224. 面试官：给多次 repeat 结果，计算 mean/std，并标记是否可发布。

答：题意是聚合多次 trial。思路是先过滤状态不是 success 或 performance_valid 为 false 的 trial；如果剩余数量不足，就不发布性能数字；否则计算 mean/std 和样本数。

边界是空列表、只有一次、某次 OOM、某次环境被污染、单位不一致。验证时要包含 mixed status。

```python
valid = [x.value for x in trials if x.status == "success" and x.performance_valid]
if len(valid) < min_repeats:
    return {"publish": False, "reason": "not enough valid repeats"}
mean = sum(valid) / len(valid)
std = variance(valid) ** 0.5
```

复杂度 O(n)。这个题和 MiniTrainBench repeat/provenance 很贴，回答时要说“不能为了凑数把 smoke 当正式结果”。

### Q225. 面试官：给每个 rank 的 step time，聚合 min、p50、max 和 straggler ratio。

答：题意是把 rank 级指标变成训练 step 指标。思路是排序后取 min/p50/max，straggler ratio 可以用 `max / p50` 或 `max / mean`。同步训练里总 step time 更接近 max。

边界是 rank 缺失、0 或负数、p50 定义、偶数 rank。验证时构造均衡和单个慢 rank 两种数据。

```python
times = sorted(rank_times)
p50 = times[len(times) // 2]
return {
    "min": times[0],
    "p50": p50,
    "max": times[-1],
    "straggler_ratio": times[-1] / p50,
}
```

复杂度 O(r log r)，r 是 rank 数。如果只要 p50 也可以用选择算法，但面试里排序更清晰。

### Q226. 面试官：写一个 checkpoint 可恢复性检查函数。

答：题意是判断一个 checkpoint 目录能不能用于 resume。思路是检查目录存在、READY 存在、metadata 存在、strategy/world size/config fingerprint 匹配、每个 rank 的 RNG 或 shard 文件齐全。

边界是 latest 指向不存在目录、半成品目录、旧 schema 缺字段、world size 不一致。验证时分别构造 READY 缺失、metadata mismatch 和完整 checkpoint。

```python
def is_recoverable(path, expected):
    if not exists(path / "READY"):
        return False
    meta = load_json(path / "metadata.json")
    if meta["world_size"] != expected.world_size:
        return False
    return meta["fingerprint"] == expected.fingerprint
```

复杂度主要是 O(r)，r 是 rank 文件数。面试里要强调“目录存在不等于可恢复”。

### Q227. 面试官：怎么实现 checkpoint atomic publish？

答：题意是写 checkpoint 时避免半成品被 resume。思路是先写到临时目录，所有文件写完并校验后，再写 READY，并更新 latest。更新 latest 也要尽量用临时文件替换，避免写一半。

边界是中途失败、多个 writer、对象存储没有 rename 语义、rank 0 写 metadata 前其他 rank 已完成。验证时模拟写到一半 crash，确认 latest 仍指向旧 READY。

```python
tmp = ckpt_root / f".tmp_step_{step}"
final = ckpt_root / f"step_{step:08d}"
write_shards(tmp)
write_metadata(tmp)
touch(tmp / "READY")
rename(tmp, final)
atomic_write_text(ckpt_root / "latest", final.name)
```

这个题就是 MiniTrainBench checkpoint 生命周期的 coding 版。生产对象存储里可能要改成 manifest commit marker。

### Q228. 面试官：写一个带指数退避的 retry。

答：题意是对临时失败的操作重试，比如拉镜像、读对象存储、请求监控接口。思路是捕获可重试异常，等待 `base * 2^attempt` 加一点 jitter；不可重试异常直接抛出。

边界是最大次数、超时时间、幂等性、哪些异常可重试。验证时构造前两次失败第三次成功，以及永久失败。

```python
for i in range(max_attempts):
    try:
        return op()
    except TransientError:
        if i == max_attempts - 1:
            raise
        sleep(base * (2 ** i) + random_jitter())
```

训练 Infra 里不能对所有操作盲目 retry。比如 checkpoint publish 不是天然幂等，必须先设计状态机。

### Q229. 面试官：写一个 evidence manifest，记录文件 SHA256。

答：题意是扫描结果文件，生成可追溯 manifest。思路是按路径排序，逐块读取文件算 SHA256，记录相对路径、hash、大小、mtime 或来源信息。排序很重要，否则 manifest 每次输出顺序不稳定。

边界是大文件、符号链接、文件读到一半变化、隐藏临时文件。验证时对同一目录重复运行，manifest 应该稳定。

```python
for path in sorted(root.rglob("*.json")):
    if path.name.startswith(".tmp"):
        continue
    manifest.append({
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
    })
```

复杂度 O(total bytes)。MiniTrainBench 的 evidence manifest 就是为了防止结果来源不可追溯。

### Q230. 面试官：写一个检测外部 GPU 进程污染的函数。

答：题意是正式 benchmark 前判断 GPU 是否被占用。思路是解析 `nvidia-smi` 或 NVML 输出，检查每张 GPU 上的 compute process、显存占用和 PID 是否属于当前 runner。

边界是 `nvidia-smi` 不存在、MIG、权限不足、短时间进程闪现、当前进程自身占用。验证时用 fake output 覆盖空闲、外部占用和解析失败。

```python
for proc in gpu_processes:
    if proc.pid not in allowed_pids and proc.used_mb > threshold_mb:
        return {"ok": False, "pid": proc.pid, "gpu": proc.gpu}
return {"ok": True}
```

这个题要讲清楚：检测失败或发现污染时，正式性能实验应该拒绝发布，而不是继续跑然后假装可信。

### Q231. 面试官：给 rank 心跳事件，找每个 rank 最后一次心跳。

答：题意是典型 hash map 题，输入是 `(rank, timestamp, step)` 事件流，输出每个 rank 的最新状态。思路是用 dict 按 rank 覆盖，只有 timestamp 更新才替换。

边界是乱序事件、重复事件、缺 rank、timestamp 相同。验证时构造 rank 0 正常、rank 1 心跳停止、rank 2 乱序上报。

```python
latest = {}
for e in events:
    old = latest.get(e.rank)
    if old is None or e.ts > old.ts:
        latest[e.rank] = e
```

复杂度 O(n)。这个题能自然延伸到 hang detector：某个 rank 太久没心跳，就可能卡住或退出。

### Q232. 面试官：比较连续训练和 resume 训练的 loss 曲线是否一致。

答：题意是 two pointers 题。两个序列可能 step 对齐，也可能缺少某些 step。思路是两个指针按 step 前进，step 相同就比较 loss 差值，step 不同就报告缺失或多余。

边界是浮点误差、step 重复、序列乱序、warmup step 是否纳入比较。验证时准备 exact match、少一步、loss 超 tolerance 三类。

```python
i = j = 0
while i < len(a) and j < len(b):
    if a[i].step == b[j].step:
        assert abs(a[i].loss - b[j].loss) <= tol
        i += 1; j += 1
    elif a[i].step < b[j].step:
        report_missing_in_b(a[i]); i += 1
    else:
        report_extra_in_b(b[j]); j += 1
```

复杂度 O(n + m)。这和 checkpoint exact verify 的思想一致：不要只看“能继续跑”，还要比较轨迹。

### Q233. 面试官：从 profiler events 中找 top K 最慢 op。

答：题意是 heap 题。输入是很多 event，每个有 name、duration、rank；输出耗时最高的 K 个。思路是维护大小为 K 的小根堆，避免全量排序。

边界是 K 大于事件数、duration 单位不一致、同名 op 是否聚合、每 rank 分开还是全局合并。验证时构造小样本，并确认 top K 顺序正确。

```python
heap = []
for e in events:
    item = (e.duration_us, e.name, e.rank)
    heappush(heap, item)
    if len(heap) > k:
        heappop(heap)
top = sorted(heap, reverse=True)
```

复杂度 O(n log k)。面试里可以补一句：真正 profiler 分析还要看 step breakdown 和时间线，不是 top K 就能证明瓶颈。

### Q234. 面试官：合并 GPU 预约区间，判断某个 job 能不能插进去。

答：题意是 interval 题。每个已有预约有 `[start, end)` 和 GPU 数，新 job 也有时间区间和 GPU 需求。简化版可以先按时间排序，合并重叠区间，检查重叠窗口里的已用 GPU 是否超过 quota。

边界是端点相等是否重叠、不同 GPU 型号、跨节点拓扑、抢占优先级。验证时用完全不重叠、端点相接、部分重叠、超过 quota 四类。

```python
events = []
for s, e, g in reservations:
    events.append((s, +g))
    events.append((e, -g))
events.sort()
used = 0
for t, delta in events:
    used += delta
    if overlaps_new_job(t) and used + need > quota:
        return False
```

这个题是调度题的简化版。真实训练平台还要考虑 gang scheduling 和网络拓扑。

### Q235. 面试官：写一个简单 GPU quota 并发控制器。

答：题意是同时有多个 job 请求 GPU，控制总使用量不超过 quota。思路是维护当前 used GPU 和等待队列；如果资源足够就启动，否则排队；job 结束时释放资源并尝试唤醒队列。

边界是公平性、优先级、饥饿、大 job 长期等不到资源、失败释放资源。验证时用多个 job 交错 submit/finish。

```python
def submit(job):
    if used + job.gpus <= quota:
        start(job)
        used += job.gpus
    else:
        queue.append(job)

def finish(job):
    used -= job.gpus
    drain_queue()
```

复杂度取决于队列策略。面试里要说明这只是 toy scheduler，生产还需要 gang scheduling、preemption 和 checkpoint 配套。

### Q236. 面试官：写一个 hang detector。

答：题意是通过心跳和 step 进度判断训练是否 hang。思路是记录最新 global_step、最后更新时间、各 rank 心跳。如果超过阈值没有 step 推进，并且 rank 心跳或日志停在同步阶段，就标记 suspect hang。

边界是正常长 step、checkpoint 保存耗时、profiler 慢、数据加载慢、误报。验证时用正常推进、长 checkpoint、真实无进度三种序列。

```python
if now - last_step_update > timeout:
    if all_recent_heartbeats_stuck_same_stage(ranks):
        return "suspect_hang"
    return "slow_or_noisy"
return "healthy"
```

面试里要强调 detector 只能给 suspect，不应直接误杀 job。下一步要拉 stack、NCCL 日志和节点健康。

### Q237. 面试官：检测 collective 调用序列是否不一致。

答：题意是每个 rank 上报 collective 序列，比如 `all_reduce(shape)`、`all_gather(shape)`，判断是否有 rank 走了不同控制流。思路是按 index 比较所有 rank 的 op type 和 shape。

边界是某个 rank 缺日志、异步日志乱序、shape 表示不同、合法的不同 group collective。验证时构造 rank 1 少一次 all-reduce、rank 2 shape 不同。

```python
for i in range(max_len):
    expected = seqs[0][i]
    for rank, seq in seqs.items():
        if i >= len(seq) or seq[i] != expected:
            report_mismatch(rank, i, expected, seq[i] if i < len(seq) else None)
```

复杂度 O(r * c)。这个题很贴 NCCL hang 排查，因为 collective 不一致常常表现成等待或 timeout。

### Q238. 面试官：写一个 config fingerprint 函数。

答：题意是从配置里选出影响训练语义的字段，生成稳定 digest。思路是只选模型、策略、world size、precision、batch、seq length、optimizer、scheduler、seed 等语义字段，排序后 JSON 序列化，再 hash。

边界是不该把 output path、log level、run name 这种非语义字段放进去；浮点、默认值和旧 schema 要处理一致。验证时改 log path fingerprint 不变，改 world size fingerprint 变化。

```python
semantic = {k: cfg[k] for k in SEMANTIC_KEYS}
payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
return sha256(payload.encode()).hexdigest()
```

MiniTrainBench 的 config fingerprint 就是为了避免错误 resume。coding 时要能解释“为什么不是所有字段都进 fingerprint”。

### Q239. 面试官：实现 checkpoint retention，只保留最近 N 个 READY checkpoint。

答：题意是清理旧 checkpoint，但不能删除半成品或唯一恢复点。思路是扫描 `step_*` 目录，只保留有 READY 的目录，按 step 排序，删除最旧的多余 READY；临时目录可以按单独规则清理。

边界是 N=0 表示全部保留、latest 指向的目录不能删、目录名解析失败、删除中途失败。验证时构造 READY、非 READY、latest 和不同 N。

```python
ready = sorted(find_ready_checkpoints(root), key=lambda x: x.step)
if keep_last == 0:
    return
for ckpt in ready[:-keep_last]:
    if ckpt.name != latest_name:
        remove_tree(ckpt.path)
```

复杂度 O(c log c)，c 是 checkpoint 数。面试里要说清楚：清理策略不能破坏恢复能力。

### Q240. 面试官：现场 coding 不会完整写出来时，怎么组织答案？

答：题意是考临场工程表达。我的顺序是先复述输入输出，再给核心数据结构，然后写主路径代码，最后补边界和测试。训练 Infra coding 题通常不是追求炫技，而是看你能不能写出可靠的小工具。

比如日志解析题，我会先定义输入是一行行日志，输出是按 step/rank 聚合的结构；数据结构用 dict；主路径逐行 parse；边界讲乱序、缺字段、NaN；测试讲正常、异常和多 rank。这样即使代码没完全写完，面试官也能看到工程思路。

如果完全卡住，我会把问题降级成最小版本先跑通，再说明如何扩展。不要沉默，也不要把不确定的实现硬说成对。

## 十六、MoE / Megatron / 大模型并行专项面经

这一节专门补 MoE、Megatron-style training 和大模型混合并行。它不是纯模型结构背诵，而是面向训练 Infra：通信、显存、rank 拓扑、checkpoint、optimizer、profile 和故障边界。MiniTrainBench 已经有 toy MoE routing、all-to-all benchmark、toy TP/SP correctness 和 Megatron compatibility smoke；完整 MoE layer、完整 Megatron runtime、生产级 sharded checkpoint 和正式 Megatron performance benchmark 仍然是明确边界。

### 16.1 MoE 深挖

### Q241. 面试官：MoE layer 的完整前向路径你怎么讲？

答：我会按 router、dispatch、expert compute、combine 四步讲。每个 token 先经过 router 得到 expert 分数，再选 top-k expert；然后 token 会按目标 expert 所在 rank 重新打包，通过 all-to-all dispatch 到对应 rank；每个 rank 对本地 expert batch 做 MLP 计算；最后 expert output 再 all-to-all 回原 token owner，并按 router weight combine 回原顺序。

这个路径难在它既有模型语义，又有系统语义。router 决定负载，dispatch 决定通信，expert compute 决定 kernel 效率，combine 决定恢复 token 顺序。MiniTrainBench 只做了 toy top-1 routing、capacity/overflow 统计和 all-to-all 通信语义，不是完整 MoE 训练层。

### Q242. 面试官：top-1 routing 和 top-2 routing 在系统上有什么差别？

答：top-1 每个 token 只发给一个 expert，通信和计算相对简单；top-2 每个 token 会发给两个 expert，再按权重合并，通常训练信号更丰富，但 dispatch token 数、expert compute 和 combine 成本都会上升。

系统上 top-2 会放大 all-to-all 压力，也会让 capacity、overflow 和 load balance 更敏感。面试里我不会只说 top-2 效果可能更好，而是会补一句：它让 runtime 要处理更多 token 副本、更多 buffer 和更复杂的 backward 路径。

### Q243. 面试官：capacity factor 为什么是 MoE 系统题，而不只是算法题？

答：capacity factor 决定每个 expert 在一个 batch 中最多接收多少 token。它看起来是 router 超参，但系统上直接影响显存峰值、expert compute 时间、buffer 大小和 step 抖动。

capacity 太小，overflow token 变多，可能损伤训练质量；capacity 太大，热门 expert 会拿到更多 token，某些 rank 变成 straggler。MiniTrainBench 的 toy route 会统计 capacity、overflow 和 expert load，能证明我理解这层语义，但没有验证完整 MoE 收敛。

### Q244. 面试官：overflow token 应该怎么处理？

答：常见策略包括丢弃、走 residual 路径、fallback 到备用 expert，或者在 capacity 更大时接收。每种策略都有 tradeoff：丢弃简单但可能损伤质量；fallback 更复杂；提高 capacity 会增加显存和计算峰值。

训练 Infra 面试里，我会先问目标是稳定性还是质量。如果是稳定跑超大规模训练，capacity 和 overflow 策略要让显存峰值可控；如果更关注效果，就要评估 overflow 对 loss 和下游指标的影响。MiniTrainBench 只展示 overflow 统计，不宣称有真实质量结论。

### Q245. 面试官：load balancing loss 解决什么问题？

答：它鼓励 router 不要把 token 都分给少数 expert，让 expert 负载更均匀。系统上它的作用是减少 straggler，让 all-to-all split 更均衡，也让每个 expert 的 batch size 更稳定。

但它不是越大越好。负载均衡项太强，可能压制 router 的表达能力；太弱，系统会被热门 expert 拖慢。面试里我会把它讲成质量和系统效率之间的桥，而不是只背“让 expert 均衡”。

### Q246. 面试官：MoE token packing 为什么需要 prefix sum？

答：router 选完 expert 后，每个 token 的目标 rank 和 expert 都不一样。为了调用 all-to-all，通常要先统计每个 destination rank 要收多少 token，再用 prefix sum 计算每个目标段在连续 buffer 中的写入位置，最后把 token pack 到连续 buffer。

这一步很关键，因为 all-to-all 更适合按 split 发送连续 buffer。实现错了会出现 token 顺序错、shape 错、rank 间 split 对不上，最后可能表现成结果错或 collective hang。MiniTrainBench 的 toy routing 只是展示这个语义，没有实现高性能 token permutation kernel。

### Q247. 面试官：all-to-all dispatch 和 combine 为什么要成对出现？

答：dispatch 是把 token 从原 owner rank 发到 expert owner rank，combine 是把 expert output 发回原 token owner，并恢复到原 token 顺序。MoE 不是把 token 发出去就结束，因为 Transformer 后续层仍需要按原 batch/sequence 位置继续计算。

backward 也要沿着这条路径反向传播：expert output 的梯度要回到 expert，router weight 和 token hidden state 的梯度也要正确聚合。所以 all-to-all 不只是 forward 通信，它会影响完整训练图。

### Q248. 面试官：MoE backward 路径难在哪里？

答：难点是梯度要沿 dispatch/combine 的反路径回去。expert MLP 的参数梯度在 expert owner rank 上产生；token hidden state 的梯度要回到原 token owner；router 权重的梯度还依赖 combine 时的 expert output。

系统上还要保证 backward 的 split、token permutation 和 forward 对齐。只要 forward pack 顺序或 metadata 记录不完整，backward 就可能对错 token。MiniTrainBench 当前没有完整 MoE backward，所以面试时只能讲原理和边界。

### Q249. 面试官：grouped GEMM 在 MoE 里解决什么？

答：每个 expert 接收到的 token 数可能不同，如果逐 expert 单独跑小 GEMM，kernel launch 多、矩阵小、GPU 利用率差。grouped GEMM 的思路是把多个不同形状或同类 expert 计算组织到更高效的 kernel 调度中，提升吞吐。

但 grouped GEMM 不是只改一个 kernel。前面 token packing、expert batch metadata、capacity 和 padding 都会影响它的效率。MiniTrainBench 没有实现 grouped GEMM，所以不能把 MoE 通信 demo 说成完整性能优化。

### Q250. 面试官：expert placement 会影响 MoE 性能吗？

答：会。expert 放在哪些 rank 或节点上，会影响 token dispatch 的跨卡、跨节点通信路径。热门 expert 如果集中在某些慢链路节点上，就容易产生通信热点和 straggler。

训练平台里 expert placement 需要考虑拓扑、负载均衡、故障隔离和扩缩容。MiniTrainBench 的 all-to-all equal/uneven split 只能模拟通信形态，不能覆盖真实 placement 策略。

### Q251. 面试官：MoE profile 应该重点看哪些指标？

答：我会看 router load distribution、overflow rate、每个 expert 的 token 数、dispatch all-to-all 时间、combine all-to-all 时间、expert compute 时间、rank max/p50、buffer size 和端到端 step time。

只看 all-to-all 平均带宽不够，因为 MoE 的瓶颈往往来自不均衡。一个 rank 收到特别多 token，就会拖住同步 step。MiniTrainBench 的 uneven all-to-all 和 profiler rank spread 可以作为面试证据，但不能替代完整 MoE layer profile。

### Q252. 面试官：如果被问“你 MoE 没做完整，为什么还讲 MoE”，怎么答？

答：我会直接承认没有实现完整 MoE layer、grouped GEMM 或真实 MoE backward。然后说明我做的是 MoE 训练 Infra 最核心的一段通信证据：router 如何造成 token dispatch，expert parallel 为什么依赖 all-to-all，equal/uneven split 为什么能暴露负载不均。

这个回答的重点是边界清楚。完整 MoE 训练当然还需要 expert compute、aux loss、backward、kernel 和收敛验证；但我已经把 MoE 与 DDP/FSDP/ZeRO 不同的通信本质讲清楚了。

### 16.2 Megatron-style 并行深挖

### Q253. 面试官：Megatron 里的 process group 为什么复杂？

答：因为一个 rank 可能同时属于 TP group、PP group、DP group、EP group，甚至 CP group。不同 group 负责不同通信语义：TP 做层内聚合，PP 做相邻 stage P2P，DP 做副本间同步或 optimizer shard，EP 做 expert dispatch。

复杂点不只是建几个 group，而是 rank mapping 要全局一致。所有 rank 必须用相同顺序创建 group，否则某些 rank 在等一个不存在的通信伙伴，就会启动 hang。MiniTrainBench 当前只有默认 world group，所以不能表达完整 Megatron 拓扑。

### Q254. 面试官：`world_size = TP * PP * DP` 这个公式够用吗？

答：基础 dense 模型里常用这个公式，但真实 Megatron-style 训练还可能有 CP、EP、expert data parallel、virtual pipeline 等维度。引入 MoE 后，普通 DP group 和 expert parallel group 可能不是同一个概念。

面试里我会先用简单公式讲清楚，再补边界：公式只是资源分解入口，不代表 group mapping 已经正确。真正实现要有 rank generator、维度顺序和每类 group 的生命周期管理。

### Q255. 面试官：为什么 group 初始化顺序不一致会 hang？

答：collective 和 process group 创建本身都要求相关 rank 参与。如果 rank 0 先创建 TP group，rank 1 先创建 PP group，就可能两边都在等不同的伙伴，最后表现为启动阶段 hang。

排查时我会先看所有 rank 的拓扑配置、group 创建顺序、rank mapping 和日志。MiniTrainBench 有多机/NCCL 诊断思路，但没有完整 group manager，所以这个问题属于 Megatron 读码和岗位知识储备。

### Q256. 面试官：Column Parallel 和 Row Parallel 分别放在哪些层？

答：Column Parallel 按输出维切权重，常用于 attention 的 QKV projection 和 MLP 的 up/gate projection。Row Parallel 按输入维切权重，常用于 attention output projection 和 MLP down projection，对 partial output 做 reduce。

这样设计的好处是中间大的 hidden 或 intermediate activation 可以保持分片，减少不必要 all-gather。MiniTrainBench 的 toy TP MLP 验证了 column 后接 row 的 forward/backward correctness，但没有实现完整 Transformer block。

### Q257. 面试官：vocab parallel 和 vocab-parallel cross entropy 解决什么？

答：大模型的 vocab 很大，如果每个 rank 都保存完整 embedding 或 logits，显存和通信压力很高。vocab parallel 把 vocab 维切到不同 TP rank 上，LM head 和 embedding 都可以按 vocab shard 处理。

cross entropy 也要配套并行化，否则需要 all-gather 完整 logits。真实实现会做分布式 max、sum exp 和目标 token 所在 shard 的 loss 计算。MiniTrainBench 没做 vocab parallel，所以面试里只能作为 Megatron 读码知识讲。

### Q258. 面试官：Sequence Parallel 和 Context Parallel 有什么区别？

答：Sequence Parallel 通常和 TP 搭配，把部分 activation 沿 sequence 维切分，减少每个 TP rank 的激活显存，常见于 LayerNorm、Dropout 等路径。Context Parallel 更偏长上下文训练，把 attention 的 context 或序列维跨 rank 切分，目标是处理更长 sequence。

两者都和长序列显存相关，但通信模式和实现范围不同。MiniTrainBench 有 toy sequence parallel correctness，没有 context parallel，所以不能把两者都说成项目实现。

### Q259. 面试官：Pipeline Parallel 的 1F1B schedule 怎么讲？

答：1F1B 是 pipeline parallel 中常见的调度方式，经过 warmup 填充 pipeline 后，每个 stage 尽量交替做一个 forward 和一个 backward，最后 cooldown 排空剩余 micro-batch。它相比先全部 forward 再全部 backward，可以降低 activation 驻留时间。

难点是 stage 间 P2P、activation 保存、loss stage、backward 依赖和 micro-batch 调度都要正确。MiniTrainBench 没有 PP schedule，所以我会明确说这是 Megatron 读码和外部 smoke 的知识。

### Q260. 面试官：virtual pipeline parallelism 解决什么？

答：virtual pipeline 会把一个物理 pipeline stage 再切成多个 virtual chunk，让流水线更细，减少 bubble 或改善 stage balance。它能提高利用率，但会增加调度复杂度、通信次数和 activation 管理难度。

面试里我会把它和 bubble 联系起来：stage 越多、micro-batch 越少，bubble 越明显；virtual PP 是降低 bubble 的一种工程手段，不是免费加速。MiniTrainBench 没实现 virtual PP。

### Q261. 面试官：micro-batch 数和 pipeline bubble 有什么关系？

答：micro-batch 数越多，pipeline 填充和排空的固定成本越容易被摊薄，bubble 占比越低；micro-batch 数太少，很多 stage 会在 warmup/cooldown 阶段闲着。

但 micro-batch 不是越多越好，因为它会影响 activation、通信次数、optimizer step 频率和 global batch 语义。Megatron compatibility smoke 里记录了 PP bubble proxy，但没有 pipeline trace，所以不能声称观察到了完整 idle 行为。

### Q262. 面试官：stage balance 为什么重要？

答：PP 里每个 stage 持有不同层，如果某个 stage 计算特别慢，整个 pipeline 都会被它拖住。即使 bubble 理论上很小，stage 不均衡也会导致吞吐差。

stage balance 要考虑层计算量、embedding/loss 所在 stage、attention/MLP 比例、激活大小和通信路径。MiniTrainBench 没做 layer partition，因此只能在面试里讲设计思路，不能说项目验证了 stage balance。

### Q263. 面试官：Megatron distributed optimizer 和 ZeRO/FSDP 怎么对比？

答：Megatron distributed optimizer 通常在 data-parallel ranks 间切分 optimizer state 和 gradient buffer，配合 contiguous parameter/gradient buffer、reduce-scatter 和参数 all-gather。ZeRO/FSDP 也做状态分片，但工程边界和 runtime 管理方式不同。

对比时不要只背名字，要说清楚参数、梯度、optimizer state 哪些 replicated、哪些 sharded，参数 gather 发生在什么时候，checkpoint 如何保存。MiniTrainBench 有 FSDP 和 ZeRO adapter 证据，但没有 Megatron distributed optimizer 的内部实现。

### Q264. 面试官：contiguous gradient buffer 有什么价值？

答：它把很多小参数的梯度组织进连续 buffer，减少大量小 collective 和内存碎片，也更方便做 bucket、reduce-scatter 和通信 overlap。大模型里参数很多，如果每个 tensor 单独通信，固定延迟会很高。

代价是实现复杂：参数到 buffer 的映射、梯度写入、optimizer shard、checkpoint 和 dtype 转换都要维护。MiniTrainBench 当前没有自定义 contiguous buffer，这是 Megatron/DeepSpeed 这类框架更完整的地方。

### Q265. 面试官：overlap-grad-reduce 和 overlap-param-gather 分别是什么？

答：overlap-grad-reduce 是 backward 过程中某些梯度 ready 后，就尽早 reduce 或 reduce-scatter，试图和后续 backward compute 重叠。overlap-param-gather 是 forward 前提前 gather 后面层需要的参数，试图隐藏参数通信。

难点是 overlap 只有 trace 能证明，不能只看 op 总耗时。MiniTrainBench 的 profiler 章节一直强调 `key_averages()` 不能证明 overlap，这个原则同样适用于 Megatron optimizer 和 FSDP prefetch。

### Q266. 面试官：Megatron distributed checkpoint 比普通 checkpoint 难在哪？

答：普通 checkpoint 可以假设同配置、同 world size 恢复；Megatron distributed checkpoint 要记录 sharded state、shard placement、TP/PP/DP/EP 拓扑、rank mapping，甚至支持目标拓扑变化时的 resharding。

MiniTrainBench 的 checkpoint 强项是 READY/latest、RNG、fingerprint 和 exact verify，但它要求同 strategy、同 world size 和同 rank mapping。面试里要清楚说：项目验证了恢复语义，不支持 Megatron 级别的跨拓扑 reshard。

### Q267. 面试官：为什么 gradient accumulation 不等于 Pipeline Parallel？

答：gradient accumulation 是同一个模型副本上分多次 micro-batch 累积梯度，再做一次 optimizer step。Pipeline Parallel 是把模型层切到不同 stage，让 micro-batch 在 stage 间流动。

两者都涉及 micro-batch，但系统语义完全不同。GA 主要影响通信频率和显存，PP 还涉及 P2P、bubble、stage balance 和 activation 传递。MiniTrainBench 做了 GA，不做 PP，这是面试里必须讲清楚的边界。

### Q268. 面试官：Megatron compatibility smoke 和完整 benchmark 的差距在哪里？

答：compatibility smoke 证明某个拓扑能完成 forward/backward/optimizer，不代表性能数字可信。完整 benchmark 还需要固定 NGC/TE 环境、独占 GPU、足够 warmup 和 measured steps、repeat、provenance、trace 或指标，以及不被 fallback kernel 影响。

MiniTrainBench 的 Megatron 外部实验现在是 core 版本固定、五组 8 卡拓扑跑通，但 fallback 环境且 GPU 非独占，performance_valid 是 false。这个边界越早说清楚越可信。

### 16.3 相关训练系统高频补充

### Q269. 面试官：FlashAttention 在训练 Infra 里解决什么？

答：FlashAttention 主要优化 attention 的内存访问和中间激活物化，减少 HBM 读写，提高 attention 性能，尤其在长序列场景明显。它不是改变 attention 数学结果，而是改变计算组织方式。

系统上要关注硬件支持、dtype、sequence length、mask、dropout、kernel 版本和数值一致性。MiniTrainBench 没有集成 FlashAttention，所以面试里我会把它作为 kernel/系统知识储备，不写成项目能力。

### Q270. 面试官：Transformer Engine 和 fused kernel 为什么重要？

答：大模型训练里很多性能来自 fused kernel 和专用库，比如 fused LayerNorm、fused attention、FP8/BF16 路径和高效 GEMM。Transformer Engine 这类栈能把多个小 op 融合，减少 kernel launch 和内存读写。

这也是为什么 Megatron fallback 环境不能和 NGC/TE 正式环境横向比较。MiniTrainBench 在文档里把 fallback 性能标为 invalid，就是因为 kernel profile 是实验变量。

### Q271. 面试官：`CUDA_DEVICE_MAX_CONNECTIONS=1` 为什么会被 Megatron 提到？

答：这个环境变量会影响 CUDA work queues 和通信计算调度。Megatron 某些 overlap 或 sequence parallel 场景会要求特定设置，让 kernel 和 NCCL 的执行顺序更可控。

我不会把它背成万能优化参数。正确回答是：它和具体 Megatron 版本、并行策略、kernel 栈有关，需要看官方要求和 trace 验证。MiniTrainBench 没有基于这个变量做正式性能结论。

### Q272. 面试官：long-context training 会放大哪些 Infra 问题？

答：长上下文会放大 attention 计算、activation 显存、KV/attention 中间状态、通信和 checkpoint 时间。即使参数量不变，sequence length 增大也可能让 activation 成为主要显存瓶颈。

因此会引入 FlashAttention、activation recompute、sequence/context parallel 和更细的 profiler 分析。MiniTrainBench 默认 sequence length 较小，不能用它直接证明长上下文训练性能，但可以用它解释方法论。

### Q273. 面试官：Context Parallel 和 Sequence Parallel 什么时候值得考虑？

答：当 sequence length 很长，单卡或单个 TP group 上的 activation/attention 内存压力过大时，才会考虑这些更复杂的序列维切分。它们的目标是把长序列压力分摊到多个 rank。

代价是额外通信、attention mask 处理、位置编码、dropout/RNG 和 checkpoint 更复杂。面试里我会把它们作为长上下文训练的扩展知识，不说 MiniTrainBench 已经覆盖。

### Q274. 面试官：Megatron、DeepSpeed、FSDP 工程栈怎么选？

答：我会先看团队已有生态、模型规模、并行需求和 checkpoint 语义。如果主要是 PyTorch 原生、希望较低接入成本，FSDP 是自然选择；如果团队已有 DeepSpeed 配置和 ZeRO 经验，可以用 ZeRO；如果需要成熟的 TP/PP/EP/CP 混合并行，Megatron-style 栈更完整。

选择不是谁高级谁赢，而是看 runtime 生命周期、状态管理、debug 成本、性能目标和团队维护能力。MiniTrainBench 把 FSDP 放在主 Trainer、ZeRO 做 adapter、Megatron 做外部 smoke，正是这个工程取舍的体现。

### Q275. 面试官：大模型并行 profile 时为什么不能只看 tokens/sec？

答：tokens/sec 是结果指标，不能告诉你瓶颈在哪。大模型混合并行里，瓶颈可能来自 TP 层内 collective、PP bubble、EP all-to-all、DP reduce-scatter、optimizer buffer、data input 或某个 fused kernel。

所以我会拆 step breakdown、rank spread、collective 类型、消息大小、kernel 时间线和内存峰值。MiniTrainBench 的 profiler 和 comm benchmark 是小规模证据，真实 Megatron 还需要更完整 trace。

### Q276. 面试官：为什么大模型训练里“环境锁定”特别重要？

答：因为 PyTorch、CUDA、NCCL、cuDNN、Transformer Engine、APEX、driver、容器 digest 和 kernel fusion flag 都可能影响性能甚至能否启动。换一个镜像，可能从 fused kernel 变成 unfused fallback，性能数字就不可比。

MiniTrainBench 的 Megatron smoke 把 fallback 环境和非独占 GPU 标成 performance_valid=false，就是为了避免把 compatibility 结果包装成正式性能。训练 Infra 里，环境也是实验协议的一部分。

### 16.4 高压追问模板

### Q277. 面试官：你没实现完整 Megatron，为什么还敢讲 Megatron？

答：我会说我不是把 Megatron 当成项目已实现能力，而是把它作为训练 Infra 的读码和对照材料。我的项目内部做了 DDP/FSDP/ZeRO、checkpoint、profiler、MoE all-to-all、toy TP/SP；外部固定 Megatron 版本跑了 compatibility smoke，并写了工程 case study。

所以我能讲清楚 Megatron-style 并行的核心问题：process group、TP/PP、distributed optimizer、sharded checkpoint 和性能证据边界。但我不会说自己复刻了 Megatron runtime。

### Q278. 面试官：MoE 没有完整训练，你怎么证明你理解 MoE？

答：我会从系统路径证明，而不是硬说做了完整层。MoE 的训练 Infra 核心是 router 造成 token 重新分布，expert parallel 依赖 all-to-all，capacity 和 overflow 决定负载与稳定性，uneven split 会制造 straggler。

MiniTrainBench 做了 toy routing 和 all-to-all equal/uneven benchmark，能把这条路径讲清楚。完整 MoE 还需要 grouped GEMM、backward、aux loss、kernel 和收敛验证，这是我明确没做的边界。

### Q279. 面试官：compatibility smoke 和 performance benchmark 最大区别是什么？

答：compatibility smoke 回答“能不能启动、能不能完成一小段 forward/backward/optimizer”；performance benchmark 回答“在严格实验协议下到底多快、多省显存、是否稳定”。前者看功能路径，后者看可信数字。

正式性能需要独占硬件、固定镜像、repeat、warmup、measured steps、provenance 和污染检测。MiniTrainBench 文档里反复区分 smoke、performance_valid 和 public report，就是为了避免混淆这两个层级。

### Q280. 面试官：如果下一步补 MoE/Megatron，你会先补什么？

答：如果只选一个方向，我会先补 topology/process group manager 和更完整的 Megatron-style evidence，因为它是 TP/PP/EP/CP 组合的地基。没有稳定的 group mapping，后面模型切分、checkpoint 和 profiler 都容易碎片化。

MoE 方向我会先补 token packing metadata、dispatch/combine correctness 和简单 expert MLP，再考虑 grouped GEMM 和 backward。无论补哪条线，我都会保持当前项目原则：先做最小正确闭环，再做性能结论，最后才说生产能力。

## 十七、FlagCX / Transformer Engine / 异构通信与低精度训练专项面经

这一节补两个更贴近大模型训练 Infra 的库级话题：FlagCX 代表异构芯片通信和 x-CCL 抽象，Transformer Engine 代表低精度训练、fused kernel 和 Megatron 常见加速栈。MiniTrainBench 当前没有集成 FlagCX，也没有集成 TE/FP8；它们在这里作为岗位知识储备和未来扩展方向。能绑定到本项目的证据，主要是 NCCL collective benchmark、MoE all-to-all、Megatron fallback 标记 invalid、环境锁定、provenance 和 profiler。

参考链接：[FlagCX](https://github.com/flagos-ai/FlagCX)、[FlagCX Tests](https://docs.flagos.io/projects/FlagCX/en/latest/testing.html)、[Transformer Engine Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)、[Transformer Engine PyTorch API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)。

### 17.1 FlagCX / 异构通信库面经

### Q281. 面试官：FlagCX 是什么，为什么训练 Infra 会关心它？

答：我会把 FlagCX 理解成面向多芯片、多厂商场景的通信库或通信抽象层。传统训练里，如果都是 NVIDIA GPU，NCCL 是最常见选择；但真实集群可能有不同芯片、不同厂商 CCL、不同互联路径，训练框架希望上层 collective 语义尽量统一。

训练 Infra 关心它，是因为通信库直接影响 all-reduce、all-gather、reduce-scatter、all-to-all 这些核心路径。MiniTrainBench 当前只实测 NCCL/Gloo，不支持 FlagCX；所以面试里我会说这是异构通信方向的知识储备，不说成项目能力。

### Q282. 面试官：FlagCX 和 NCCL、HCCL、RCCL、CNCL 这些 x-CCL 是什么关系？

答：我的理解是，NCCL、HCCL、RCCL、CNCL 这类库更偏具体芯片或厂商生态下的 collective 实现；FlagCX 这类跨芯片通信库的价值，是在这些 native x-CCL 之上或旁边，提供更统一的通信接口和跨芯片数据交换能力。

回答时要谨慎，不要说“FlagCX 替代所有 CCL”。更准确是：同构场景优先用厂商最强 native backend；异构或跨生态场景需要一个能协调不同 backend、P2P、IPC、RDMA 和 fallback 的抽象层。

### Q283. 面试官：为什么异构芯片训练需要统一通信抽象？

答：因为上层训练框架关心的是 collective 语义，比如 all-reduce 梯度、all-gather 参数、all-to-all token dispatch，而不是每次都手写不同芯片的通信细节。如果没有统一抽象，同一个训练策略在不同硬件上要写多套路径，debug 和 benchmark 都会变得非常碎。

但统一抽象也有代价。它可能隐藏 backend 差异，让性能问题更难定位。所以好的通信抽象必须把 backend、拓扑、消息大小、fallback 状态和性能指标暴露出来，不能只给一个“成功完成”的布尔值。

### Q284. 面试官：FlagCX 里 host API 和 device API 可以怎么理解？

答：host API 可以理解成由 CPU/host 侧发起的高层 collective 调用，接近训练框架常见的通信入口；device API 更偏设备侧或 kernel 级通信能力，适合做更底层的 P2P、IPC、RDMA 或细粒度性能测试。

面试里不需要硬背具体函数名，重点是讲清楚两层关注点不同：host API 看上层 collective 是否易接入、易 benchmark；device API 看更低层的数据路径、同步和性能上限。FlagCX 文档里也把 performance test 分成 host API 和 device API 两类。

### Q285. 面试官：device-buffer IPC 和 device-buffer RDMA 解决什么问题？

答：它们解决的是设备内存之间高效数据交换的问题。IPC 更偏同节点或近距离设备间共享/访问 device buffer，RDMA 更偏跨节点直接访问远端内存，目标都是减少不必要的 host 拷贝和中间转发。

训练系统里，这会影响跨芯片 P2P 和 collective 的效率。比如 all-to-all 或 all-gather 如果频繁经过 host staging，延迟和带宽都会受影响。但具体收益必须通过 benchmark 验证，不能只听名字就认为一定更快。

### Q286. 面试官：FlagCX 和 PyTorch distributed 的边界在哪里？

答：PyTorch distributed 提供训练框架层的 process group、collective API 和 DDP/FSDP 等能力；FlagCX 这类通信库更像底层或中间层 backend，负责具体数据如何跨设备、跨节点、跨芯片移动。

如果要接入 MiniTrainBench，我会优先把它放进 communication benchmark 或 process group backend 层，而不是直接改 Trainer 语义。先证明 all-reduce/all-gather/reduce-scatter/all-to-all 能跑、指标能记录，再讨论训练 runtime 接入。

### Q287. 面试官：怎么 benchmark FlagCX 这种通信库？

答：我会沿用通信 benchmark 的基本方法：固定 world size、设备、拓扑、dtype、message size 和 warmup/measured iterations，分别测 all-reduce、all-gather、reduce-scatter、all-to-all 的 latency、bandwidth、p50/p95、rank spread 和失败率。

异构场景还要额外记录 backend 选择、芯片组合、链路类型、是否跨节点、是否 fallback、是否经过 host staging。MiniTrainBench 现在已有 NCCL collective benchmark 和 MoE all-to-all，可以作为扩展 FlagCX benchmark 的模板。

### Q288. 面试官：NCCL 和 FlagCX benchmark 怎么公平比较？

答：首先要明确比较目标。如果是同构 NVIDIA 集群，NCCL 是强 baseline；FlagCX 如果走 NCCL backend 或封装层，就不能把封装开销和原生 NCCL 混着解释。如果是异构集群，NCCL 本身可能不能覆盖所有设备组合，那比较目标就变成“是否能统一跑起来，以及代价是多少”。

公平比较需要固定 message size、rank 数、拓扑、dtype、warmup、repeat 和独占环境，并记录 backend 路径。不能只拿一个 tokens/sec 说谁更快，必须说明它跑的是同构、异构、native backend 还是 fallback。

### Q289. 面试官：FlagCX/CCL 通信慢了怎么排查？

答：我会先看 backend 是否符合预期，再看 rank/world size、拓扑、网卡、驱动/runtime、环境变量和 message size。然后用最小 collective benchmark 复现，区分是某个 op 慢、某种 size 慢，还是只有训练 step 慢。

如果是异构场景，还要看是否发生 fallback、是否经过 host 拷贝、某类芯片是否成为 straggler、不同 backend 的同步语义是否一致。排查顺序和 NCCL 类似，但异构通信更需要把实际路径暴露出来。

### Q290. 面试官：FlagCX/CCL 出现 hang，最可能是什么原因？

答：常见原因仍然是 collective 参与者不一致：rank 数不一致、某些 rank 没进入同一个 collective、op type 或 tensor shape 不一致、group mapping 错、某个 backend 初始化失败但上层没感知。异构场景还可能有某个设备 backend 卡住，其他 rank 在等待。

我的处理方式是先加 timeout 和分阶段日志，记录每个 rank 即将进入的 op、shape、group 和 backend。MiniTrainBench 的 collective sequence mismatch coding 题和 NCCL 诊断思路可以迁移到 FlagCX，但项目没有真实 FlagCX hang 证据。

### Q291. 面试官：FlagCX 对 MoE all-to-all 有什么意义？

答：MoE expert parallel 最核心的通信就是 token dispatch/combine 的 all-to-all。如果训练集群是异构芯片，MoE 的 all-to-all 会更加难，因为 token 可能跨不同 backend、不同带宽和不同延迟路径流动。

FlagCX 这类库的价值，是给异构 all-to-all 提供更统一的数据交换能力。但 MoE 性能最终还取决于 router load balance、token packing、expert placement、grouped GEMM 和 straggler。MiniTrainBench 只验证了 NCCL all-to-all equal/uneven split，不代表 FlagCX MoE 性能。

### Q292. 面试官：如果把 FlagCX 接入 MiniTrainBench，你会先做什么？

答：我会先做最小通信 backend adapter，而不是一上来改训练主循环。第一步扩展 `comm` benchmark，让它能选择 NCCL/Gloo/FlagCX 类 backend，输出相同 schema；第二步记录 backend、设备、拓扑、版本、fallback 状态和环境变量；第三步只发布 collective 结果，不急着宣称训练加速。

等 communication evidence 稳定后，再考虑 DDP/FSDP 或 MoE path 的集成。这样符合项目原则：先验证通信 primitive，再把它放进训练 runtime。

### 17.2 Transformer Engine 面经

### Q293. 面试官：Transformer Engine 是什么？

答：Transformer Engine 可以理解成 NVIDIA 面向 Transformer 模型训练和推理的加速库，提供 FP8/更低精度支持、fused kernels 和 PyTorch/JAX 等接口。它不是单纯替换一个 Linear，而是把低精度、scaling、GEMM、LayerNorm、attention 和并行训练里的很多优化打包起来。

MiniTrainBench 当前没有集成 TE。项目里和 TE 相关的证据，是 Megatron fallback 缺少 TE/APEX fused capability 时，把结果标为 compatibility smoke，而不是 performance benchmark。

### Q294. 面试官：为什么 TE 不是普通 PyTorch Linear 的简单替换？

答：因为 TE 的价值不只是模块 API 相似，而是背后有低精度 recipe、scaling metadata、fused kernel、Tensor Core 路径、并行训练支持和环境依赖。把 `nn.Linear` 换成 `te.Linear` 可能只是第一步，真正性能还取决于 dtype、shape、硬件、kernel 选择和是否进入 fused path。

所以 benchmark TE 要记录 GPU 架构、TE 版本、CUDA/cuDNN、Megatron 配置、是否 FP8、是否 fused attention/LayerNorm、是否 fallback。否则“换了 TE 没变快”或“TE 更快”都说不清楚。

### Q295. 面试官：FP8 和 BF16/FP16 怎么对比？

答：BF16 指数范围大，训练稳定性通常比较好；FP16 精度和范围更敏感，常需要 loss scaling；FP8 进一步降低存储和计算成本，能提升吞吐、降低显存和带宽压力，但需要更复杂的 scaling 和硬件支持。

训练 Infra 里 FP8 不是简单把 dtype 改掉。要记录 scaling recipe、amax history、哪些 op 用 FP8、哪些保持 BF16/FP32、checkpoint 是否保存必要状态，以及恢复后数值路径是否一致。MiniTrainBench 目前只覆盖 BF16/FP32。

### Q296. 面试官：FP4、MXFP8、NVFP4 这类更低精度怎么讲才不虚？

答：我会把它们讲成更激进的低精度格式，通常依赖更新硬件和更严格的软件栈，目标是进一步降低内存和计算成本。但它们比 FP8 更需要关注数值误差、scaling、kernel 支持和适用场景。

面试里不应该装成自己生产用过。更稳的说法是：我知道 TE 文档已经覆盖 FP8，并在更新硬件上支持 MXFP8、NVFP4 这类路径；我当前项目没有验证这些格式，所以只能讲原理、约束和实验设计。

### Q297. 面试官：amax history 和 scaling recipe 是什么？

答：低精度训练里，需要把高精度 tensor 映射到 FP8 等低精度范围。amax 是一段时间内观察到的最大绝对值，scaling recipe 决定怎么用这些 amax 计算 scale，比如当前 scaling、delayed scaling 或 per-tensor/per-channel 等策略。

它们会影响数值稳定性和性能。scale 太小容易溢出，太大又损失精度。checkpoint/resume 时如果 FP8 scaling 状态没有恢复，训练轨迹可能漂移。所以如果 MiniTrainBench 未来接 TE/FP8，scaling metadata 必须进入状态和 provenance。

### Q298. 面试官：DelayedScaling 怎么口述？

答：DelayedScaling 可以理解成用历史 amax 来决定当前或下一段计算的 scale，而不是每个 tensor 都即时重新估计。这样可以减少同步和统计开销，也让 scale 更新更平滑。

但它带来一个状态问题：历史 amax buffer 和 scale 本身成为训练状态的一部分。恢复 checkpoint 时，如果这些状态缺失，就算模型权重和 optimizer 一样，后续 FP8 数值路径也可能不一致。

### Q299. 面试官：TE 的 fused LayerNorm / fused attention / operation fuser 解决什么？

答：它们主要减少 kernel launch、减少中间 tensor 物化、降低 HBM 读写，并利用专用 kernel 提高吞吐。Transformer block 里有很多小 op，如果都走普通 PyTorch eager 路径，调度和内存开销会很明显。

但 fused kernel 的收益取决于 shape、硬件、dtype 和软件版本。小模型、短序列或 fallback 环境下不一定更快。MiniTrainBench 的 Megatron fallback 经验正好说明：kernel capability 是实验变量，必须记录。

### Q300. 面试官：fuse weight gradient accumulation 是什么？

答：普通训练里，weight gradient 计算和 gradient accumulation 可能是分开的路径；fuse weight gradient accumulation 是把权重梯度计算和累积更紧密地融合，减少额外读写和 kernel 调度开销。

这对大模型和 micro-batch accumulation 有价值，但也会改变 profiler 里的 op 形态。验证时不能只看 loss，还要看 grad 是否一致、显存峰值、step time 和 checkpoint/resume 状态。当前 MiniTrainBench 没有 TE 这个融合路径。

### Q301. 面试官：TE 和 Megatron 的关系怎么讲？

答：Megatron 是大模型并行训练框架，负责 TP/PP/DP/EP、distributed optimizer、schedule 和 checkpoint 等系统逻辑；TE 提供 Transformer 层、低精度和 fused kernel 能力。两者经常配合：Megatron 管并行和训练流程，TE 管很多高性能 Transformer kernel。

所以 Megatron fallback 环境缺 TE/APEX 时，能跑通 topology 不代表性能可比。MiniTrainBench 已经把这种情况标成 performance_valid=false，这个回答能体现我懂“框架能跑”和“性能栈完整”不是一回事。

### Q302. 面试官：TE 和 TP/SP/MoE 有什么关系？

答：TE 不只是单卡 kernel，它也会和 TP、SP、MoE 这类并行模式交互。比如 TE Linear 可能参与 tensor parallel；sequence parallel 会影响 LayerNorm/Dropout/activation 的布局；MoE 里可能需要 grouped linear、router、expert parallel 和 fused kernels 配合。

系统难点是低精度状态、并行 shard、通信和 kernel metadata 要对齐。MiniTrainBench 有 toy TP/SP 和 toy MoE routing，但没有 TE 并行集成，所以只能讲这个交互关系和未来扩展方向。

### Q303. 面试官：小模型上 TE 为什么可能不明显变快？

答：小模型的 GEMM 可能不够大，kernel launch、框架调度、数据准备和通信固定开销占比高，fused kernel 的优势不一定能摊开。低精度还可能引入 cast、scale、amax 统计等额外开销。

所以 TE benchmark 要选合适的模型规模、sequence length、batch size 和测量窗口。MiniTrainBench 的小模型结果一直强调不能外推大模型，这个原则同样适用于 TE。

### Q304. 面试官：TE 接入后 checkpoint/resume 要多保存什么？

答：除了 model、optimizer、scheduler、TrainState 和 RNG，还要考虑 FP8 scaling state、amax history、recipe 配置、TE module 状态、并行 shard metadata，以及软件版本。否则恢复后可能功能上能跑，但数值轨迹不一致。

MiniTrainBench 现在没有 TE，所以 checkpoint v3 不包含这些状态。如果未来加 TE，我会先把它纳入 `checkpoint verify` 的比较范围，再发布 FP8 resume 结论。

### 17.3 高压追问模板

### Q305. 面试官：你项目没集成 FlagCX，为什么讲 FlagCX？

答：我会直接说 FlagCX 是岗位知识储备和未来扩展方向，不是项目已实现能力。我的项目已经把 NCCL collective、MoE all-to-all、rank spread、provenance 和 communication benchmark 做出来了，所以我可以从这些已有证据自然迁移到异构通信库应该怎么接、怎么测、怎么排障。

换句话说，我不是说“我用过 FlagCX 训练大模型”，而是说“我理解通信库在训练 Infra 里的位置，并知道要先从 collective benchmark 和 backend evidence 做起”。

### Q306. 面试官：你项目没集成 TE/FP8，怎么证明你懂？

答：我会承认 MiniTrainBench 只覆盖 BF16/FP32，没有 TE/FP8 实现。然后把理解落到系统层：TE 涉及 fused kernel、FP8 scaling、amax history、硬件支持、Megatron 集成和 checkpoint 状态；这些点会影响性能、数值稳定和恢复一致性。

项目证据是我在 Megatron fallback 上没有混淆 compatibility smoke 和 performance benchmark。fallback 缺 TE/APEX fused capability 时，我把性能标为 invalid，这说明我理解 TE 是性能栈的一部分。

### Q307. 面试官：NCCL 和 FlagCX benchmark 怎么避免自欺欺人？

答：先固定实验协议，再公开实际 backend。必须记录设备类型、拓扑、rank 数、message size、dtype、warmup、repeat、是否独占、backend 版本和 fallback 状态。对于同构 NVIDIA，NCCL 是直接 baseline；对于异构设备，要说明 NCCL 是否适用，FlagCX 是否走 native x-CCL、IPC/RDMA 或 fallback。

如果这些没记录，就不能说 FlagCX 比 NCCL 快或慢。最多说某个配置下 smoke 成功。MiniTrainBench 的 performance_valid 和 provenance 设计正是为了避免这种自欺欺人。

### Q308. 面试官：TE fallback、NGC/TE 正式环境和 performance_valid 怎么区分？

答：fallback 环境可以证明代码路径能跑，但缺少 TE/APEX fused kernels、Transformer Engine 低精度路径或官方推荐环境时，性能数字不能和 NGC/TE 正式环境横向比较。正式 performance benchmark 需要固定 base digest、TE import 成功、fused kernel capability 明确、GPU 独占、repeat 和完整 provenance。

所以我会把 fallback 结果标成 compatibility，不展示或不发布吞吐结论。这个原则已经在 MiniTrainBench 的 Megatron smoke 里使用过。

### Q309. 面试官：如果给 MiniTrainBench 加 FlagCX 和 TE，你先做哪个？

答：如果目标是训练 Infra 岗位展示，我会先加 FlagCX 风格的 communication backend adapter，因为它能复用现有 comm benchmark，风险小，边界清楚。先把 all-reduce、all-gather、reduce-scatter、all-to-all 的 backend/provenance/schema 做好，再讨论接入训练主循环。

TE 我会作为第二步，因为它会影响模型模块、dtype、scaling state、checkpoint 和 profiler，需要更完整的 correctness 和 resume 验证。两者都不应该一上来追性能数字，先要证明语义和证据链可靠。

### Q310. 面试官：SeedInfra 面试里怎么把异构通信和低精度训练讲得不虚？

答：我会把它们都落到训练系统 contract。异构通信不是背库名，而是讲 collective 语义、backend 路径、拓扑、fallback 和 benchmark 公平性；低精度训练不是背 FP8 名字，而是讲 scaling state、fused kernel、硬件支持、数值稳定、checkpoint 和 provenance。

最后再主动说边界：MiniTrainBench 当前没集成 FlagCX/TE，但它已经有通信 benchmark、profiler、环境锁定和 performance_valid 这些地基。我的准备方向是先把这些系统方法讲清楚，再补具体库的实战。
