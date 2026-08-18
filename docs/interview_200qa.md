# MiniTrainBench 200 道面试题和答案

这份文档按主题整理，适合快速背诵和复盘。

## 一、项目定位
1. 这个项目是做什么的？答：做一个最小分布式训练 runtime 和可复现实验套件。
2. 为什么叫 MiniTrainBench？答：Mini 表示范围克制，Bench 表示重点是训练 benchmark。
3. 它解决的核心问题是什么？答：训练状态、同步、恢复、性能和故障边界。
4. 它和普通训练脚本有什么区别？答：它有完整状态管理、恢复逻辑和验证流程。
5. 为什么说它是 runtime，不只是脚本？答：因为它管理训练生命周期，不只是跑一次。
6. 项目主要面向什么场景？答：单节点 pretraining infra。
7. 为什么不直接做大而全框架？答：先把核心契约做实，范围更清楚，结果更可信。
8. 为什么不用真实数据集？答：为了消除下载、预处理和数据噪声。
9. 项目的数据是什么？答：deterministic synthetic token。
10. synthetic data 的好处是什么？答：可复现、稳定、无外部依赖。
11. 项目最重要的价值是什么？答：把训练系统做成能验证、能解释、能复查。
12. 这个项目是不是生产框架？答：不是，边界是最小 runtime 和证据套件。
13. 这个项目是不是 toy demo？答：不是，它有真实分布式训练和故障恢复证据。
14. 为什么要做成“最小”？答：这样每个行为都能讲清楚、测清楚、证清楚。
15. 项目最像哪类岗位需求？答：训练 Infra、分布式训练、性能工程。
16. 项目有没有 GPU 实测？答：有，做了 1/2/4/8 卡实测。
17. 项目有没有 CPU CI？答：有，CPU/Gloo smoke 负责保底。
18. 项目有没有 provenance？答：有，结果里记录 commit、镜像和命令。
19. 为什么 provenance 重要？答：没有 provenance 的结果不能当正式证据。
20. 一句话怎么总结项目？答：一个能跑、能恢复、能解释的最小分布式训练系统。

## 二、核心架构
1. `Trainer` 做什么？答：负责训练循环、状态推进和结果汇总。
2. `TrainingConfig` 做什么？答：保存所有训练超参数和运行配置。
3. `TrainState` 做什么？答：保存 step、token、seed 和恢复信息。
4. `StepMetrics` 做什么？答：记录每一步的时间、loss、lr 和 grad norm。
5. `TrainingStrategy` 做什么？答：隔离 DDP/FSDP 的包装和同步差异。
6. 为什么要抽 strategy？答：让并行策略和训练主循环解耦。
7. `DDPStrategy` 的作用是什么？答：封装 DDP 包装和默认同步语义。
8. `FSDPStrategy` 的作用是什么？答：封装 FSDP 包装和分片语义。
9. 为什么 DDP 和 FSDP 不能写死在 Trainer 里？答：它们的通信和状态管理不同。
10. `CheckpointManager` 做什么？答：统一管理保存、恢复、READY 和 latest。
11. `SyntheticTokenIterator` 做什么？答：生成确定性的训练 batch。
12. `LearningRateScheduler` 做什么？答：管理 constant 或 cosine 学习率。
13. 为什么要保存 `config_fingerprint`？答：防止拿错配置的 checkpoint 恢复。
14. 为什么要保存 `resumed_from`？答：方便追踪恢复链路。
15. 为什么要保存 `tokens_seen`？答：便于确认训练进度和恢复位置。
16. 为什么要记录 `global_step` 和 `micro_step`？答：分别表示外层步数和累积内步数。
17. 为什么要记录 `learning_rate`？答：验证 scheduler 是否按预期推进。
18. 为什么要记录 `grad_norm`？答：看训练是否稳定、是否发生裁剪。
19. 为什么要记录 `max_cuda_memory_mb`？答：评估显存压力。
20. 项目架构的主线是什么？答：状态、同步、恢复、验证、证据。

## 三、数据与可复现
1. synthetic data 是怎么生成的？答：按 seed、global_step 和 rank 生成。
2. 为什么要把 rank 放进 seed 体系里？答：避免每个 rank 生成完全一样的数据。
3. 为什么要把 global_step 放进去？答：保证每一步拿到不同 batch。
4. 为什么不用随机从数据集里抽？答：那会引入额外不稳定因素。
5. 为什么 repeat 要独立初始化？答：这样每个 trial 才是独立样本。
6. 为什么不能把 repeat 当连续窗口？答：那会混入优化器状态和缓存影响。
7. repeat 统计看什么？答：mean、std、min、max。
8. 为什么要报 std？答：只看单次结果不够可信。
9. 为什么要 warmup？答：排除启动和缓存冷态影响。
10. 为什么要测量时同步 GPU？答：避免异步计时失真。
11. 为什么 step time 不能只看 rank 0？答：同步训练受最慢 rank 限制。
12. 为什么跨 rank 要取 max？答：这更接近真实 step 时间。
13. 为什么 loss 常常取 mean？答：loss 是数值指标，不是同步瓶颈。
14. 为什么显存通常取 max？答：看最坏情况，避免低估风险。
15. 为什么 tokens/sec 是全局指标？答：训练吞吐取决于全局 batch。
16. 为什么要固定 base image？答：避免环境漂移影响结果。
17. 为什么要固定 commit？答：方便回溯代码版本。
18. 为什么要记录完整命令？答：方便重跑和审计。
19. 为什么旧结果不能直接和新结果混比？答：provenance 不同，结论可能失效。
20. 可复现性的核心是什么？答：数据、状态、环境、命令都要固定。

## 四、DDP、FSDP 和梯度累积
1. DDP 是什么？答：每个 rank 持有完整模型，用 all-reduce 同步梯度。
2. FSDP 是什么？答：把参数、梯度和 optimizer state 分片保存。
3. DDP 的优点是什么？答：实现简单，通常吞吐更高。
4. FSDP 的优点是什么？答：显存更省，更适合大模型。
5. 为什么 FSDP 不一定更快？答：要多做参数 all-gather 和梯度 reduce-scatter。
6. 什么是 gradient accumulation？答：多个 micro-batch 累加后再更新一次。
7. DDP 累积时最容易犯什么错？答：每个 micro-batch 都同步梯度。
8. 正确做法是什么？答：前面 micro-batch 用 `no_sync()`，最后一个再同步。
9. 为什么 `no_sync()` 要包住 forward 和 backward？答：DDP 的 reducer 状态依赖 forward。
10. FSDP 为什么默认不走 DDP 那套末步同步？答：因为它可能拉高峰值显存。
11. DDP 的 `auto` 实际解析成什么？答：`last`。
12. FSDP 的 `auto` 实际解析成什么？答：`every`。
13. 为什么 FSDP 默认是 `every`？答：优先显存安全。
14. 什么是 global batch size？答：local batch 乘 world size 再乘 accumulation 步数。
15. 为什么 loss 要除以 accum steps？答：保持梯度尺度不变。
16. 什么是 gradient clipping？答：把过大的梯度范数截住。
17. 为什么要记录 grad norm？答：判断训练是否健康。
18. 为什么要记录 clipped_steps？答：判断裁剪是否真的发生。
19. 为什么 DDP 小模型通常更快？答：通信开销相对更小。
20. DDP/FSDP 的核心区别一句话怎么说？答：DDP 重吞吐，FSDP 重显存。

## 五、Checkpoint 和 Resume
1. checkpoint 里保存了什么？答：model、optimizer、scheduler、TrainState 和 RNG。
2. 为什么不能只保存 model？答：optimizer 和 scheduler 也会影响后续轨迹。
3. 为什么要保存 RNG？答：dropout 等随机路径会改变恢复后的结果。
4. READY 标记是什么意思？答：这个 checkpoint 已经完整发布。
5. latest 是什么？答：指向最新 READY checkpoint 的入口。
6. 为什么要先写临时目录？答：避免半成品被误当成可恢复点。
7. 保存顺序是什么？答：临时目录、写数据、写 READY、原子替换、更新 latest。
8. 为什么要做 retention？答：控制 checkpoint 数量和磁盘占用。
9. prune 什么时候做？答：新 checkpoint 成功发布后再做。
10. 为什么恢复时要检查 READY？答：没有 READY 就不算正式发布。
11. 为什么要检查 strategy？答：不同策略的状态语义不一样。
12. 为什么要检查 precision？答：精度不同，状态和数值语义可能不同。
13. 为什么要检查 world size？答：分片大小和恢复布局必须一致。
14. 为什么要检查 config fingerprint？答：防止拿错配置的 checkpoint。
15. 什么是 exact resume？答：恢复后和连续训练状态一致。
16. v3 checkpoint 新增了什么？答：scheduler 和每 rank RNG。
17. 旧 checkpoint 怎么兼容？答：保留 legacy fingerprint 和降级逻辑。
18. 旧 checkpoint 能做到 exact resume 吗？答：不一定，通常会降级为功能性恢复。
19. 为什么 FSDP 的 checkpoint 校验更难？答：状态里可能有分片 tensor。
20. checkpoint 的核心目标是什么？答：从“能恢复”升级到“能证明恢复正确”。

## 六、故障恢复和稳定性
1. 故障恢复解决什么问题？答：训练中断后如何回到正确状态。
2. 项目做过什么真实故障测试？答：真实 worker `SIGKILL`。
3. `SIGKILL` 测试想验证什么？答：launcher 是否非零退出，checkpoint 是否不变。
4. 恢复模式是什么？答：manual restart。
5. 自动弹性恢复做了吗？答：没有。
6. 为什么要明确说没有自动恢复？答：避免把边界说大了。
7. 什么是半成品 checkpoint？答：目录存在，但还没 READY。
8. 怎么发现半成品 checkpoint？答：扫描 latest 和 READY。
9. config mismatch 怎么处理？答：直接拒绝恢复。
10. 为什么不能悄悄改配置恢复？答：那样训练语义会变。
11. NaN 怎么处理？答：所有 rank 一起 fail-fast。
12. 为什么要 all-rank fail-fast？答：避免不同 rank 走向不同状态。
13. 非有限值检测看什么？答：loss 和 grad norm。
14. 检测到非有限值后先做什么？答：清空梯度，再中止。
15. 为什么要记录 deterministic 标记？答：说明恢复是否能精确复现。
16. deterministic 不成立意味着什么？答：可以继续训练，但不能保证完全一致。
17. 为什么要保留故障脚本？答：让故障边界可重现、可审计。
18. 故障处理最重要的原则是什么？答：宁可拒绝恢复，也不要错误恢复。
19. 为什么说 checkpoint 比 crash 更重要？答：恢复点错了，后面全错。
20. 稳定性的核心是什么？答：把异常变成可预期行为。

## 七、Profiler 和性能
1. 为什么要做 Profiler？答：为了知道慢在哪里。
2. Profiler 和 train benchmark 为什么分开？答：避免 profiling 开销污染吞吐结果。
3. Profiler 输出什么？答：每 rank Chrome trace 和摘要报告。
4. step breakdown 里有什么？答：data、forward/backward、optimizer、step time。
5. 为什么要看 top ops？答：找最耗时的算子。
6. 为什么要看 collective？答：分布式训练常常慢在通信。
7. 什么是 straggler ratio？答：最慢 rank 和中位数的比值。
8. 为什么要看 p50 而不是只看 mean？答：p50 更能代表典型 rank。
9. 为什么要记录每 rank 诊断？答：分布式瓶颈常常不是平均值能看出来的。
10. 为什么不能只看 key_averages？答：它看不到跨 stream 的时间关系。
11. 怎么判断计算和通信是否 overlap？答：看 Chrome trace 或 Perfetto。
12. 为什么要看 CPU op？答：Python 或 host 开销也可能是瓶颈。
13. 为什么要看 CUDA op？答：核心计算和通信都在 GPU 侧。
14. 为什么要看 memory？答：吞吐高不代表显存安全。
15. Profiler 和 benchmark 的分工是什么？答：benchmark 给数字，profiler 给原因。
16. 什么时候用单次 profiler 就够？答：主要是定位问题，不是做统计。
17. 什么时候应该看 repeat 统计？答：做正式性能结论时。
18. 为什么 tokens/sec 不能单独看？答：要结合 step time 和显存一起看。
19. 为什么 DDP 和 FSDP 的 profiler 结果差别明显？答：FSDP 有更多参数和梯度通信。
20. 性能工程的核心方法是什么？答：先量，再拆，再解释。

## 八、通信和 MoE
1. all-reduce 在训练里做什么？答：同步 DDP 梯度。
2. all-gather 在训练里做什么？答：FSDP 前向前取完整参数。
3. reduce-scatter 在训练里做什么？答：FSDP 把梯度分片回去。
4. all-to-all 在训练里做什么？答：MoE 里按 token 重新分发。
5. 为什么要测 all-to-all？答：MoE 的瓶颈经常在这里。
6. equal split 是什么？答：每个 peer 收发相同大小的数据。
7. uneven split 是什么？答：每个 peer 收发大小不同。
8. 为什么 uneven 更贴近真实 MoE？答：router 往往会造成负载不均。
9. 通信 benchmark 主要看什么？答：延迟和带宽。
10. 为什么先看小 tensor？答：小 tensor 更容易暴露延迟。
11. 为什么再看大 tensor？答：大 tensor 更容易接近带宽上限。
12. MoE 的核心不是参数同步，而是什么？答：token dispatch 和 combine。
13. 什么是 router？答：决定 token 该去哪个 expert。
14. 什么是 capacity factor？答：每个 expert 的容量上限。
15. 什么是 overflow？答：超过容量后被丢弃的 token。
16. 什么是 load imbalance？答：不同 expert 收到的 token 数差很多。
17. 为什么 load imbalance 重要？答：最慢 expert 会拖慢整个 step。
18. 为什么要做 token packing？答：减少通信前的数据整理成本。
19. 为什么要做 combine？答：把分发出去的 token 结果还原回来。
20. MoE 的一句话总结是什么？答：核心难点是 all-to-all 和负载均衡。

## 九、Tensor Parallel、ZeRO 和 Megatron
1. ColumnParallelLinear 是什么？答：按输出维切 Linear。
2. RowParallelLinear 是什么？答：按输入维切 Linear。
3. 为什么要先讲切分语义？答：面试官更关心你是否真正理解并行方式。
4. ColumnParallelLinear 之后通常做什么？答：把各 rank 的输出拼起来。
5. RowParallelLinear 之后通常做什么？答：把 partial output 做求和。
6. toy TP check 验证什么？答：forward 和 backward 是否和单卡一致。
7. 为什么要做 toy TP MLP？答：把一层切分扩展成完整子网络。
8. 什么是 sequence parallel？答：按序列维切分激活。
9. sequence parallel 的好处是什么？答：降低激活显存。
10. sequence parallel 的代价是什么？答：额外通信和随机性管理。
11. ZeRO-2 是什么？答：分片 optimizer state 和 gradient。
12. ZeRO-3 是什么？答：连参数也分片。
13. 为什么 ZeRO 单独做 adapter？答：DeepSpeed Engine 的生命周期和核心 runtime 不一样。
14. 为什么不把 ZeRO 直接塞进 Trainer？答：会把两套状态机混在一起。
15. Megatron case study 的目的是什么？答：理解真实生产并行框架。
16. Megatron 里验证了什么？答：兼容性 smoke。
17. 为什么不把 Megatron 的结果当正式性能结论？答：环境不满足正式基准条件。
18. 什么是 TP/PP/DP？答：tensor parallel、pipeline parallel、data parallel。
19. 什么是 pipeline bubble？答：流水线阶段的空闲时间。
20. TP/ZeRO/Megatron 的核心收获是什么？答：知道不同并行方式的切分语义和边界。

## 十、结果、边界、CI 和面试表达
1. DDP 8 卡结果说明什么？答：吞吐高，适合小模型。
2. FSDP 8 卡结果说明什么？答：显存明显更省。
3. memory pressure 结果说明什么？答：FSDP 能把大模型从 OOM 拉回来。
4. 为什么要做 repeat=3？答：避免单次短跑误导结论。
5. 为什么要报 mean/std？答：让结果更稳健。
6. 为什么要记录 provenance？答：让结果可审计。
7. 为什么要做 CPU/Gloo CI？答：GPU 成本高，CI 也要守住核心逻辑。
8. CPU/Gloo CI 覆盖什么？答：forward/backward、collective、checkpoint、report。
9. 为什么不把所有验证放到 GPU 上？答：太贵，也不利于持续集成。
10. 项目明确没做什么？答：RLHF、GRPO、推理服务、编译器方向。
11. 为什么没有把多机做成主线？答：没有稳定多机资源，就很难给出可信证据。
12. 为什么要把项目边界说清楚？答：面试里更可信，也更专业。
13. 这个项目最强的证据是什么？答：真实恢复、repeat 统计和 provenance。
14. 这个项目最强的工程点是什么？答：状态机、原子发布和验证闭环。
15. 这个项目最强的性能点是什么？答：DDP/FSDP/ZeRO 的吞吐和显存对比。
16. 这个项目最强的故障点是什么？答：checkpoint、NaN、rank crash 的边界处理。
17. 这个项目最强的通信点是什么？答：all-reduce、reduce-scatter、all-to-all 的完整覆盖。
18. 面试时最该先讲什么？答：项目定位和边界。
19. 面试时最该怎么讲 checkpoint？答：先讲原子发布，再讲精确恢复。
20. 面试时怎么收尾最稳？答：我做的是最小 runtime，但证据链是完整的。
