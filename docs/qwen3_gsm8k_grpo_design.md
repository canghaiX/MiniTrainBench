# Qwen3 + SGLang GSM8K GRPO

The local Qwen3/SGLang path is implemented and tested, but the full BF16
experiment has **not passed the policy logprob acceptance gate**. Read
[`gsm8k_grpo_validation.md`](gsm8k_grpo_validation.md) before interpreting the
results. A passing short smoke is not evidence that the complete 256 × 8
rollout is numerically aligned. No updated-policy accuracy gain is claimed.

## Data and policy contract

`scripts/prepare_gsm8k.py` explicitly downloads the official main train/test
JSONL files, validates fixed SHA-256 digests before accepting a download, selects
256 training questions with seed 2026, and preserves all 1,319 test questions.
The selected indices, source URLs, counts, MIT license and hashes are recorded
in `manifest.json`. A cached source is reused only after hash verification.
The dataset and generated artifacts are excluded from Git.

Source: [official GSM8K data card](https://huggingface.co/datasets/openai/gsm8k/blob/25fb0412dbcc2e5128322234d829fbd6ece72c7c/README.md).
This is subset training with a local full-test evaluation, not a standard
GSM8K leaderboard configuration.

InferKernelLab uses the local tokenizer chat template with thinking disabled.
The prompt requests step-by-step reasoning and a final `####` numerical answer.
Reward prefers the last marked answer, otherwise the last numeric value;
commas are removed and `Decimal` compares numeric values, so `42.00` equals
`42`. This fallback also applies to truncated responses and is an experimental
reward convention, not the official benchmark evaluation harness.

The fixed policy is temperature 0.8, top-p 1, top-k disabled, eight independent
responses per group. Per-request seeds hash the base seed and flattened completion index, avoiding
overlapping random streams for adjacent base seeds. Both
configured Qwen EOS IDs are retained with their logprobs. Native SGLang
`/generate` returns temperature-scaled untruncated policy logprobs. Model,
tokenizer, prompt-file, data-manifest and SGLang server fingerprints accompany
the artifact. The script mounts local weights; it never downloads a model.

MiniTrainBench independently checks the local model/tokenizer fingerprints,
SGLang 0.5.18 server metadata, source dataset digests, prompt groups, dataset
answers, prompt encoding and generated-text decoding. It recomputes reward
from the generated token IDs and the verified dataset answer before loading
the actor/reference. Old Llama/token-prefix artifacts retain their old schema.

## Update and acceptance

`AutoModelForCausalLM` loads only local Llama or Qwen3 checkpoints. FP32 master
parameters and BF16 autocast are used for the Qwen experiment. The existing
eager path remains the default; the reproduction script explicitly selects
SDPA because it reduced the observed SGLang/HF discrepancy. A separate frozen
reference supplies the sampled KL term.

The existing group-normalized advantages, clipped policy ratio, KL penalty,
DDP gradient averaging, gradient clipping and DCP resume remain in use.
`--microbatch-groups 1` accumulates one group at a time; weighted group losses
and DDP `no_sync` preserve the intended eight groups per rank per optimizer
step without retaining all eight groups' activations. Two ranks × eight
groups × 16 steps consume the fixed 256-group offline rollout once.

Before any optimizer update, every response token is teacher-forced with
prompt/padding excluded and EOS included. The explicit Qwen3 BF16 maximum
absolute logprob tolerance is 0.5; old Llama BF16 and FP32 tolerances remain
0.05 and 0.002. The report records max/mean/P99 error, token count, worst token,
and the fraction of initial importance ratios outside [0.8, 1.2]. Nonfinite or
out-of-tolerance values write a structured failed report and stop training.
`--preflight-only` runs the same check without an optimizer update or export.
The tolerance has not been increased to turn a failed full run into a pass.

Successful runs verify rank parameter digests, export a local HF checkpoint,
and support DCP resume. Evaluation restarts SGLang with each exported model;
the summarizer requires matching seeds, model fingerprints, dataset, tokenizer,
prompt and generation settings, a passed probe, actual parameter changes and
consistent DDP parameters before calculating three-seed mean/sample std.

## Reproduction

Install `pip install -e '.[rl]'` here and `pip install -e '.[hf]'` in
InferKernelLab. Docker, curl, local Qwen3 weights and three available A100 GPUs
are required by the reproduction script. Its fixed SGLang image reports
version 0.5.18; runtime/image metadata are saved alongside results.

```bash
OUT_ROOT=results/gsm8k_grpo_run \
INFER_ROOT=/data/InferKernelLab SERVER_GPU=7 TRAIN_GPUS=0,1 \
  bash scripts/run_gsm8k_grpo_cycle.sh
```

The script fails at a failed preflight; it does not silently train or produce
updated-model accuracy statistics. Preserve the JSON diagnostic and resolve
the numerical mismatch before claiming completion of the three-seed cycle.
This is fixed offline rollout training, not an asynchronous online RL system.
