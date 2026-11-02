# Offline GRPO Rollout and Update

This optional path demonstrates one auditable post-training update using a local Llama or Qwen3 model.
InferKernelLab samples several completions per prompt and writes a JSONL artifact. MiniTrainBench
validates schema, token ranges, finish metadata, recorded reward, model/tokenizer
fingerprints, and the initial actor logprobs before training. The policy and frozen reference
start from the same local checkpoint; DDP averages actor gradients across ranks. Prompt groups
are sharded by rank and the number of groups must divide evenly across ranks.

For each response token, the ratio is
`exp(logprob_actor - logprob_old)`, where all three policies use temperature-scaled but
untruncated logprobs. The sampled clipped objective is averaged across tokens within a response,
then across responses within a prompt group, then across prompt groups. The reference penalty is
the sampled estimator `exp(logprob_ref - logprob_actor) - (logprob_ref - logprob_actor) - 1` and
uses the same per-response/per-group weighting. Prompt and padding positions do not contribute;
an emitted EOS token does. Group rewards are centered and divided by population standard
deviation. A zero-variance group receives zero advantages and is counted in the report.

Top-p filtering affects which token is sampled but not the stored policy logprob used for the
ratio. This is an explicit approximation: it does not form an importance ratio for a truncated
behavior distribution. Both distributions are recorded in the InferKernelLab artifact so the
choice is visible. Before the first update, the trainer recomputes every actor logprob and fails
if the artifact does not match the loaded policy within the configured tolerance.

Checkpoints use the project's existing DCP manager and include optimizer, scheduler, step state,
and per-rank RNG state. Resume requires the same rollout artifact and training configuration.
Output is a regular Hugging Face model directory usable for another local rollout. The smoke
summary reports whether weights changed and whether all DDP ranks ended with identical weights;
it does not claim reward improvement.

This is intentionally not a full RLHF system: rewards are token-prefix accuracy or GSM8K numeric
exact match, the reference is frozen locally, and there is no learned reward model, critic,
asynchronous generation, TP/FSDP actor, or production data pipeline. The optional SGLang path is
described in [qwen3_gsm8k_grpo_design.md](qwen3_gsm8k_grpo_design.md); its full Qwen3 BF16
preflight currently fails the acceptance gate. The experiment exercises
the rollout-to-training artifact boundary and distributed update/resume mechanics.
