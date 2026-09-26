# Vocab-Parallel Causal LM

`minitrainbench tp lm-train` 将已有的 TP/DP Transformer Block 串成一个可训练的
causal language model。它使用 deterministic token IDs，不依赖 tokenizer 或远程数据集，
用于验证并行语义和通信边界，而不是替代完整生产训练框架。

## 参数切分

- token embedding 按 vocabulary row 切分；每个 rank 只保存
  `[vocab_start:vocab_end]`。
- embedding lookup 对不属于本 rank 的 token 置零，再通过 TP all-reduce 得到完整 hidden。
- Transformer Block 复用 Column Parallel QKV/MLP、Row Parallel output 和 replicated
  LayerNorm/residual bias。
- LM head 按 vocabulary row 切分，输出 local logits；本轮不做 tied embedding。
- process group layout 保持现有约定：`tp_rank = global_rank % tp_size`，
  `dp_rank = global_rank // tp_size`。

## Vocab-Parallel Cross Entropy

每个 rank 只接收 local logits，但 loss 需要全局 vocabulary 的 softmax。custom autograd
前向依次执行：

1. TP all-reduce global maximum；
2. 对稳定化 logits 求 local exponential sum，并 TP all-reduce 全局分母；
3. 只有持有 target vocabulary shard 的 rank 提供 target logit，再 TP all-reduce；
4. 计算 mean negative log likelihood。

反向只返回 local logits 的梯度。vocab-sharded 参数只需要 DP gradient averaging；LM head
传回 hidden 的梯度在 TP group 中聚合，避免构造完整 vocab logits。

## Correctness 口径

训练前 probe 构造相同初始化的 dense reference，并检查：

- gathered local logits 与 dense logits；
- vocab-parallel loss 与 dense cross entropy；
- embedding、LM head、Transformer 局部 shard 和 replicated 参数梯度；
- 不同 DP rank 的 deterministic batch；
- optimizer step 后 DP replica 参数一致。

FP32 使用 absolute tolerance，BF16 使用 relative L2 tolerance。benchmark 输出同时记录
TP/DP degree、P50/P95、tokens/sec 和 activation/gradient/CE collective 的通信量估算。

## 限制

当前版本只支持 MHA、untied embedding/LM head 和单机 TP/DP；不包含 checkpoint resharding、
optimizer state sharding、pipeline parallel 或 multi-node TP。复现命令见
`scripts/run_tp_dp_lm_smoke.sh`。
