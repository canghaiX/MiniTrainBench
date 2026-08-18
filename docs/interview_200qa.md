# MiniTrainBench 深度面试问答

这份文档用于面试前复盘，包含基础热身题和项目深挖题。深挖题不追求数量堆满，而是模拟真实面试里“围绕一个点继续追问”的节奏。

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
