# MiniTrainBench: Distributed Training Runtime and Evidence Suite

## Project Goal

MiniTrainBench is a compact, auditable distributed training runtime for studying the
correctness, performance, memory behavior, and failure modes of LLM training systems.
It uses a configurable GPT-like model and deterministic synthetic tokens, so experiments
do not depend on dataset downloads or preprocessing variance. The project is deliberately
smaller than a production framework: each feature exists to expose a concrete runtime
contract or produce reproducible evidence.

## What I Built

The core runtime implements a `Trainer`, explicit `TrainingConfig` and `TrainState`, and a
strategy registry for PyTorch DDP and FSDP. It supports BF16, gradient accumulation,
activation checkpointing, DDP `no_sync()` semantics, FSDP memory-safe synchronization,
constant or cosine learning-rate schedules, global gradient norm, clipping, and all-rank
non-finite fail-fast. DeepSpeed ZeRO-2 and ZeRO-3 are isolated behind a benchmark adapter,
which avoids mixing DeepSpeed Engine lifecycle and checkpoint semantics into the DDP/FSDP
runtime.

Distributed checkpointing uses PyTorch DCP plus an atomic lifecycle: write to a temporary
directory, synchronize ranks, publish `READY`, update `latest`, and prune only after a new
checkpoint is discoverable. Checkpoint format v3 stores model, optimizer, scheduler,
`TrainState`, and per-rank CPU/CUDA RNG state. `checkpoint verify` reconstructs sharded
state and compares deterministic digests. A real rank-crash experiment sends `SIGKILL` to
one worker, verifies that the launcher exits non-zero and the last READY checkpoint is
unchanged, then performs a manual restart and exact state verification.

The performance suite covers DDP/FSDP/ZeRO training, NCCL all-reduce/all-gather/
reduce-scatter, equal and uneven all-to-all, PyTorch Profiler, toy MoE routing, and toy
Tensor/Sequence Parallel correctness. Every formal JSON result records the source revision,
image ID, locked base image, launch command, Python/PyTorch/CUDA/cuDNN/NCCL/driver versions,
and output digest. Independent trials reinitialize the model and optimizer before reporting
mean, standard deviation, minimum, and maximum.

CPU/Gloo CI protects argument parsing, model forward/backward, distributed collectives,
checkpoint compatibility, repeat aggregation, profiler summaries, and Megatron log parsing
without requiring a CI GPU. GPU-only scripts then validate NCCL behavior, sharded state,
memory pressure, profiler output, and fault injection in the locked benchmark environment.

## Headline Evidence

- On 8x A100-SXM4-40GB, the repeat-3 23.2M model baseline reached
  **227,644 +/- 1,696 tokens/s with DDP** and **128,973 +/- 1,062 tokens/s with FSDP**.
- In that small-model baseline, FSDP used **172.8 MB** versus DDP's **630.1 MB**, a
  **72.6% memory reduction**, while communication overhead reduced throughput.
- In the 8-GPU memory-pressure matrix, a **2.60B parameter DDP run OOMed**, while FSDP
  completed at **14.7K tokens/s and 6.37 GB peak allocator memory**. This demonstrates the
  regime where sharding changes trainability rather than merely adding overhead.
- The rank-crash test achieved `exact_match=true` for model, optimizer, scheduler,
  `TrainState`, and every rank's RNG after manual restart. It does not claim elastic
  automatic recovery.
- Megatron-LM `core_v0.18.2` completed 8-GPU compatibility smoke for TP/PP/DP topologies
  `1/1/8`, `2/1/4`, `4/1/2`, `2/2/2`, and `1/4/2`. Those runs used an official PyTorch
  fallback while GPUs had concurrent work, so their performance fields are explicitly
  invalid and hidden from the Markdown report. NGC repeat-3 performance remains pending.

## Engineering Boundaries

MiniTrainBench is single-node and pretraining-focused. It provides multi-node launch and
NCCL diagnostic tooling, but no fabricated multi-node benchmark. Toy TP/SP and MoE demos
validate communication and correctness; they are not a replacement for Megatron's full
pipeline scheduler, distributed optimizer, expert parallel runtime, or reshardable
checkpoint system. RLHF/GRPO, inference serving, and compiler work are outside scope.

The Megatron integration follows the same boundary. Upstream source is supplied externally
and pinned to an exact tag and commit; this repository contains only the audited runner,
parser, manifests, and a code-reading case study. Compatibility evidence is separated from
performance evidence, and a formal run is rejected when GPUs are already occupied.

## Project Map

- [Main README](../README.md)
- [Runtime design](runtime_design.md)
- [Megatron engineering case study](megatron_case_study.md)
- [8-GPU profiler case study](profiler_case_study_8gpu.md)
- [Megatron compatibility report](../results/megatron_smoke/report.md)
- [Locked GPU rerun postmortem](postmortem_locked_gpu_rerun.md)

## 中文摘要

MiniTrainBench 是面向训练 Infra 的最小分布式 Runtime：实现 DDP/FSDP、ZeRO 对比、精确
checkpoint/resume、真实 rank crash、Profiler、NCCL/MoE/TP 证据链，并在 8xA100 上完成
可追溯实验。Megatron `core_v0.18.2` 五组 TP/PP/DP 拓扑已跑通兼容性 smoke；因未满足
NGC 与独占 GPU 条件，性能结果明确标为无效，正式 repeat=3 仍是项目边界。
