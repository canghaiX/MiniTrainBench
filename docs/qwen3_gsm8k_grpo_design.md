# Qwen3 HF reference GRPO and SGLang serving evaluation

The HF incremental-cache single-seed cycle is complete: 16 dual-GPU BF16 steps,
419,959 response tokens with maximum logprob error 0.0, and full-test evaluation
before/after. Both models score 677/1,319 (51.3268%): no measured EM improvement.
See the [final report](gsm8k_single_seed_result.md) for results and provenance
limitations. HF is the behavior policy; SGLang is the separate evaluation engine.

The historical local Qwen3/SGLang rollout path is implemented and tested, but its full BF16
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

Use InferKernelLab's `scripts/compare_grpo_logprobs.py` with the rollout and
failed preflight JSON to compare generation logprobs, SGLang teacher-forced
candidate scores and HF scores on identical token prefixes. The current
diagnostic shows exact agreement between stored generation and SGLang scores
for checked positions, with residual SGLang/HF forward differences above the
gate on rare tokens. See
[`gsm8k_logprob_alignment_diagnostic.json`](gsm8k_logprob_alignment_diagnostic.json).
The follow-up recheck found that full-sequence and prefix HF forwards differ
for two worst-case tokens; repeated SGLang scores were stable. The public
SGLang hidden-state response exposes only the final state. A separate
forward-hook capture found the first significant block divergence at layers
15, 17, and 7 on the three fixed worst cases; divergence grows with depth, but
the first layer is sample-dependent and does not establish a single operator
root cause. See the layerwise report in InferKernelLab. The SGLang artifact's
strict gate still fails, so it remains ineligible for optimizer updates.

InferKernelLab also provides a separate `hf_reference` correctness rollout.
It samples Qwen3 one completion at a time through Transformers `DynamicCache`,
and records the PyTorch/Transformers versions, SDPA mode, FP32 weights and
autocast dtype. MiniTrainBench selects the incremental, gradient-enabled cache
scorer only for this explicit artifact contract; old SGLang and Llama artifacts
retain their existing full-sequence path. The serial reference is intentionally
low throughput and is not a serving performance claim.

An initial A100 4-prompt/8-completion smoke passed its 1,024-token policy
preflight with max absolute error 0.0, but its 32-token cap truncated every
completion and produced no optimizer update. The expanded 8-prompt smoke then
passed all 11,332 response tokens with max error 0.0, produced within-group
reward variance, and completed one dual-GPU BF16 update. Parameters changed,
rank digests matched, and peak allocated memory was 19.7 GiB. The script now
checks these conditions. DCP resume then reached step 2, passed the same
preflight, changed the policy, and retained matching rank digests; peak
allocated memory was 23.9 GiB. The subsequent seed-2026 full 256 × 8 gate
passed for 419,959 tokens (max/mean/P99 error 0.0). Formal training completed
16 steps with changed parameters and consistent ranks; final model/TrainState
DCP readback matched the training digest. Peak allocated memory was 26.64 GiB,
training throughput 12.09 response token/s. Initial and updated full-test EM
were both 677/1,319; this single-seed experiment establishes the system cycle,
not an accuracy gain. [Final evidence](gsm8k_single_seed_result.md).

Incremental actor training processes one completion at a time and backpropagates
its equally weighted contribution immediately, preserving the mean over
completions/groups while releasing each cache graph. DDP buffer broadcasts are
disabled after initial wrapper synchronization because different completion
lengths cause different numbers of token forwards per rank; gradient reduction
still occurs on the final completion backward. The first A100 attempt exposed
both mismatched per-forward collectives and excessive graph memory; the current
path passed the two-rank update/resume smoke with a lower observed peak.

Successful runs verify rank parameter digests, export a local HF checkpoint,
and support DCP resume. Evaluation restarts SGLang with each exported model;
the summarizer requires matching seeds, model fingerprints, dataset, tokenizer,
prompt and generation settings, a passed probe, actual parameter changes and
consistent DDP parameters before calculating metrics. A single seed has an
undefined sample standard deviation (`null`), rather than a zero deviation.

## Reproduction

Install `pip install -e '.[rl]'` here and `pip install -e '.[hf]'` in
InferKernelLab. Docker, curl, local Qwen3 weights and three available A100 GPUs
are required by the reproduction script. Its fixed SGLang image reports
version 0.5.18; runtime/image metadata are saved alongside results.

```bash
OUT_ROOT=results/gsm8k_hf_reference \
INFER_ROOT=/data/InferKernelLab SERVER_GPU=7 TRAIN_GPUS=0,1 HF_REFERENCE_GPU=0 \
  bash scripts/run_gsm8k_hf_reference_cycle.sh
```

The script defaults to seed 2026 and an eight-prompt update/resume smoke. It
prepares data offline from verified local source files and runs a complete
rollout preflight before formal updates. `SEEDS="2026 2027 2028"` explicitly
enables multiple seeds; every full gate must pass before any formal update.
It fails at any failed gate and does not produce updated-model accuracy
statistics for an incomplete run. `--evaluate-only` resumes after completed
training, validating the update report before starting evaluation.

Smoke exports and checkpoints are deleted after successful resume validation;
formal training retains only the latest two DCP snapshots. Once full test
evaluation and the summary are accepted, formal checkpoints are deleted too.
The one updated model, small reports, dataset and rollout remain local. Cleanup
is restricted to named intermediate directories, rejects symlinks and writes
a JSON audit; it never deletes another experiment. Container logs are bounded
to 10 MB and the experiment container is removed on exit.
This is fixed offline rollout training, not an asynchronous online RL system.
