from __future__ import annotations

import gc
import hashlib
import json
import platform
import statistics
import time
from typing import Any

import torch

from .data import SyntheticTokenIterator
from .distributed import DistributedContext, setup_distributed
from .model import GPTConfig, MiniGPT, count_parameters
from .runtime import (
    _precision_dtype,
    _reduce_max,
    _reduce_mean,
    _summarize_repeats,
    _sync_device,
    _write_json,
)


def build_deepspeed_config(args: Any, world_size: int) -> dict[str, Any]:
    train_micro_batch_size = int(args.batch_size)
    zero_optimization = {
        "stage": int(args.zero_stage),
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 500_000_000,
        "reduce_bucket_size": 500_000_000,
    }
    if int(args.zero_stage) == 3:
        zero_optimization.update(
            {
                "stage3_prefetch_bucket_size": 50_000_000,
                "stage3_param_persistence_threshold": 100_000,
            }
        )
    return {
        "train_micro_batch_size_per_gpu": train_micro_batch_size,
        "gradient_accumulation_steps": int(args.grad_accum_steps),
        "train_batch_size": train_micro_batch_size
        * int(args.grad_accum_steps)
        * int(world_size),
        "bf16": {"enabled": args.precision == "bf16"},
        "fp16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero_optimization,
        "wall_clock_breakdown": False,
    }


def _load_deepspeed() -> Any:
    try:
        import deepspeed
    except ImportError as error:
        raise ValueError(
            "当前环境没有安装 DeepSpeed。请使用 "
            "`docker build --target gpu-deepspeed -t minitrainbench:deepspeed .` "
            "构建可选 ZeRO benchmark 镜像。"
        ) from error
    return deepspeed


def _fingerprint(values: dict[str, Any]) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_config(args: Any) -> dict[str, Any]:
    return {
        "vocab_size": args.vocab_size,
        "seq_length": args.seq_length,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }


def _initialize_engine(
    args: Any,
    context: DistributedContext,
    deepspeed: Any,
) -> tuple[Any, int, dict[str, Any]]:
    torch.manual_seed(args.seed + context.rank)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + context.rank)
    model = MiniGPT(
        GPTConfig(**_model_config(args)),
        activation_checkpointing=args.activation_checkpointing,
    ).to(context.device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    config = build_deepspeed_config(args, context.world_size)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=config,
    )
    engine.train()
    return engine, parameter_count, config


def _cleanup_engine(engine: Any | None) -> None:
    if engine is not None:
        del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_one_step(
    args: Any,
    context: DistributedContext,
    engine: Any,
    iterator: SyntheticTokenIterator,
    global_step: int,
    dtype: torch.dtype,
) -> dict[str, float]:
    _sync_device(context.device)
    started = time.perf_counter()
    data_started = started
    inputs: list[torch.Tensor] = []
    for micro_index in range(args.grad_accum_steps):
        inputs.append(
            iterator.batch_for_step(
                global_step * args.grad_accum_steps + micro_index,
                context.device,
            )
        )
    data_time = time.perf_counter() - data_started

    forward_backward = 0.0
    optimizer_step = 0.0
    last_loss = 0.0
    engine.zero_grad()
    for input_ids in inputs:
        forward_started = time.perf_counter()
        with torch.autocast(
            device_type=context.device.type,
            dtype=dtype,
            enabled=dtype != torch.float32,
        ):
            _, loss = engine(input_ids, input_ids)
        assert loss is not None
        engine.backward(loss)
        _sync_device(context.device)
        forward_backward += time.perf_counter() - forward_started
        last_loss = float(loss.detach().item())

    optimizer_started = time.perf_counter()
    engine.step()
    _sync_device(context.device)
    optimizer_step = time.perf_counter() - optimizer_started
    step_time = time.perf_counter() - started
    tokens_per_step = (
        args.batch_size * args.seq_length * args.grad_accum_steps * context.world_size
    )
    return {
        "data_time_ms": data_time * 1000,
        "forward_backward_ms": forward_backward * 1000,
        "optimizer_step_ms": optimizer_step * 1000,
        "step_time_ms": step_time * 1000,
        "tokens_per_sec": tokens_per_step / step_time,
        "loss": last_loss,
    }


def _run_trial(
    args: Any,
    context: DistributedContext,
    deepspeed: Any,
    dtype: torch.dtype,
    repeat_index: int,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    engine = None
    try:
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        engine, parameter_count, ds_config = _initialize_engine(args, context, deepspeed)
        iterator = SyntheticTokenIterator(
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
            rank=context.rank,
        )
        context.barrier()
        measured: list[dict[str, float]] = []
        total_steps = args.warmup_steps + args.steps
        for step_index in range(total_steps):
            metrics = _run_one_step(
                args,
                context,
                engine,
                iterator,
                step_index,
                dtype,
            )
            if step_index >= args.warmup_steps:
                measured.append(metrics)
        context.barrier()
        local = {
            "data_time_ms": statistics.fmean(item["data_time_ms"] for item in measured),
            "forward_backward_ms": statistics.fmean(
                item["forward_backward_ms"] for item in measured
            ),
            "optimizer_step_ms": statistics.fmean(
                item["optimizer_step_ms"] for item in measured
            ),
            "step_time_ms": statistics.fmean(item["step_time_ms"] for item in measured),
            "loss": measured[-1]["loss"],
        }
        result = {
            key: _reduce_max(value, context)
            if key != "loss"
            else _reduce_mean(value, context)
            for key, value in local.items()
        }
        result["max_cuda_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(context.device) / 1024**2)
            if context.device.type == "cuda"
            else 0.0
        )
        result["max_cuda_memory_mb"] = _reduce_max(result["max_cuda_memory_mb"], context)
        result["tokens_per_sec"] = (
            args.batch_size
            * args.seq_length
            * args.grad_accum_steps
            * context.world_size
            / (result["step_time_ms"] / 1000)
        )
        result.update(
            {
                "repeat_index": repeat_index,
                "trial_protocol": "independent_reinitialize"
                if args.repeat > 1
                else "single_run",
                "global_step": total_steps,
                "tokens_seen": (
                    args.batch_size
                    * args.seq_length
                    * args.grad_accum_steps
                    * context.world_size
                    * total_steps
                ),
            }
        )
        return result, parameter_count, ds_config
    finally:
        _cleanup_engine(engine)


def _build_result(
    args: Any,
    context: DistributedContext,
    deepspeed: Any,
    repeats: list[dict[str, Any]],
    parameter_count: int,
    ds_config: dict[str, Any],
) -> dict[str, Any]:
    summary = _summarize_repeats(repeats)
    selected = {
        "tokens_per_sec": summary["tokens_per_sec"]["mean"],
        "step_time_ms": summary["step_time_ms"]["mean"],
        "data_time_ms": summary["data_time_ms"]["mean"],
        "forward_backward_ms": summary["forward_backward_ms"]["mean"],
        "optimizer_step_ms": summary["optimizer_step_ms"]["mean"],
        "max_cuda_memory_mb": summary["max_cuda_memory_mb"]["mean"],
        "loss": repeats[-1]["loss"],
    }
    strategy = f"deepspeed_zero{args.zero_stage}"
    config = {
        "strategy": strategy,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
        "activation_checkpointing": args.activation_checkpointing,
        "grad_accum_steps": args.grad_accum_steps,
        "batch_size": args.batch_size,
        "model_config": _model_config(args),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    trial_protocol = "independent_reinitialize" if len(repeats) > 1 else "single_run"
    result: dict[str, Any] = {
        "benchmark": "training",
        "strategy": strategy,
        "world_size": context.world_size,
        "precision": args.precision,
        "device": str(context.device),
        "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
        "gradient_accumulation_steps": args.grad_accum_steps,
        "activation_checkpointing": args.activation_checkpointing,
        **selected,
        "repeat_count": len(repeats),
        "trial_protocol": trial_protocol,
        "parameters": parameter_count,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size
        * context.world_size
        * args.grad_accum_steps,
        "tokens_seen": repeats[-1]["tokens_seen"],
        "global_step": repeats[-1]["global_step"],
        "model_config": _model_config(args),
        "config": config,
        "config_fingerprint": _fingerprint(config),
        "deepspeed_config": ds_config,
        "runtime": {
            "strategy_impl": f"DeepSpeedZeRO{args.zero_stage}",
            "zero_stage": args.zero_stage,
            "trial_protocol": trial_protocol,
            "checkpoint_dir": None,
            "resume": False,
            "resume_deterministic": None,
            "resume_determinism_reason": "deepspeed_benchmark_no_checkpoint",
            "resume_path": None,
            "last_checkpoint": None,
            "latest_checkpoint": None,
            "keep_last": 0,
            "ready_checkpoints": 0,
            "global_step": repeats[-1]["global_step"],
            "tokens_seen": repeats[-1]["tokens_seen"],
            "seed": args.seed,
            "deepspeed_version": getattr(deepspeed, "__version__", "unknown"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deepspeed": getattr(deepspeed, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(context.local_rank)
            if context.device.type == "cuda"
            else None,
        },
    }
    if len(repeats) > 1:
        result["repeats"] = repeats
        result["summary"] = summary
    return result


def deepspeed_benchmark(args: Any) -> dict[str, Any] | None:
    deepspeed = _load_deepspeed()
    context = setup_distributed(
        args.backend,
        args.device,
        requires_process_group=True,
    )
    try:
        if context.device.type != "cuda":
            raise ValueError("DeepSpeed ZeRO benchmark 需要 CUDA 设备")
        dtype = _precision_dtype(args.precision, context.device)
        repeats: list[dict[str, Any]] = []
        parameter_count = 0
        ds_config: dict[str, Any] = {}
        for repeat_index in range(args.repeat):
            metrics, parameter_count, ds_config = _run_trial(
                args,
                context,
                deepspeed,
                dtype,
                repeat_index,
            )
            if context.is_main:
                repeats.append(metrics)
        if not context.is_main:
            return None
        result = _build_result(
            args,
            context,
            deepspeed,
            repeats,
            parameter_count,
            ds_config,
        )
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    finally:
        context.close()
