from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import DistributedContext, setup_distributed
from .model import GPTConfig, MiniGPT, TransformerBlock, count_parameters


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 benchmark mode requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not report BF16 support")
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


def _wrap_fsdp(model: nn.Module, context: DistributedContext, precision: str) -> nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mixed_precision = None
    if precision == "bf16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    kwargs: dict[str, Any] = {
        "auto_wrap_policy": policy,
        "mixed_precision": mixed_precision,
        "use_orig_params": True,
    }
    if context.device.type == "cuda":
        kwargs["device_id"] = context.device
    return FSDP(model, **kwargs)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_max(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    tensor = torch.tensor(value, device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.MAX)
    return float(tensor.item()) if context.is_main else value


def _reduce_mean(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    tensor = torch.tensor(value, device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size) if context.is_main else value


def _summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = ["tokens_per_sec", "step_time_ms", "max_cuda_memory_mb"]
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(repeat[metric]) for repeat in repeats]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def train(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(
        args.backend,
        args.device,
        requires_process_group=args.strategy == "fsdp",
    )
    try:
        dtype = _precision_dtype(args.precision, context.device)
        config = GPTConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        torch.manual_seed(args.seed + context.rank)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + context.rank)

        model = MiniGPT(config, activation_checkpointing=args.activation_checkpointing)
        model.to(context.device)
        parameter_count = count_parameters(model)
        if args.strategy == "ddp":
            if context.world_size > 1:
                model = DistributedDataParallel(
                    model,
                    device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                )
        elif args.strategy == "fsdp":
            model = _wrap_fsdp(model, context, args.precision)
        else:
            raise ValueError(f"unsupported strategy: {args.strategy}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
        context.barrier()

        tokens_per_step = args.batch_size * config.seq_length * args.grad_accum_steps
        repeats: list[dict[str, Any]] = []
        for repeat_index in range(args.repeat):
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)
            context.barrier()
            timings: list[float] = []
            last_loss = 0.0
            for step in range(args.warmup_steps + args.steps):
                optimizer.zero_grad(set_to_none=True)
                _sync_device(context.device)
                started = time.perf_counter()
                for _ in range(args.grad_accum_steps):
                    input_ids = torch.randint(
                        0,
                        config.vocab_size,
                        (args.batch_size, config.seq_length),
                        device=context.device,
                    )
                    with torch.autocast(
                        device_type=context.device.type,
                        dtype=dtype,
                        enabled=dtype != torch.float32,
                    ):
                        _, loss = model(input_ids, input_ids)
                    assert loss is not None
                    (loss / args.grad_accum_steps).backward()
                    last_loss = float(loss.detach().item())
                optimizer.step()
                _sync_device(context.device)
                elapsed = time.perf_counter() - started
                if step >= args.warmup_steps:
                    timings.append(elapsed)

            local_step_time = sum(timings) / len(timings)
            step_time = _reduce_max(local_step_time, context)
            mean_loss = _reduce_mean(last_loss, context)
            max_memory = (
                float(torch.cuda.max_memory_allocated(context.device) / 1024**2)
                if context.device.type == "cuda"
                else 0.0
            )
            max_memory = _reduce_max(max_memory, context)
            tokens_per_sec = tokens_per_step * context.world_size / step_time
            if context.is_main:
                repeats.append(
                    {
                        "repeat_index": repeat_index,
                        "tokens_per_sec": tokens_per_sec,
                        "step_time_ms": step_time * 1000,
                        "max_cuda_memory_mb": max_memory,
                        "loss": mean_loss,
                    }
                )

        summary = _summarize_repeats(repeats) if context.is_main else {}
        selected = {
            "tokens_per_sec": summary["tokens_per_sec"]["mean"],
            "step_time_ms": summary["step_time_ms"]["mean"],
            "max_cuda_memory_mb": summary["max_cuda_memory_mb"]["mean"],
            "loss": repeats[-1]["loss"],
        } if context.is_main else {}
        result: dict[str, Any] = {
            "benchmark": "training",
            "strategy": args.strategy,
            "world_size": context.world_size,
            "precision": args.precision,
            "device": str(context.device),
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "gradient_accumulation_steps": args.grad_accum_steps,
            "activation_checkpointing": args.activation_checkpointing,
            "tokens_per_sec": selected.get("tokens_per_sec", 0.0),
            "step_time_ms": selected.get("step_time_ms", 0.0),
            "max_cuda_memory_mb": selected.get("max_cuda_memory_mb", 0.0),
            "loss": selected.get("loss", 0.0),
            "repeat_count": args.repeat,
            "parameters": parameter_count,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * context.world_size * args.grad_accum_steps,
            "model_config": config.__dict__,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(context.local_rank)
                if context.device.type == "cuda"
                else None,
            },
        }
        if args.repeat > 1:
            result["repeats"] = repeats
            result["summary"] = summary
        if context.is_main:
            _write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
        return result if context.is_main else None
    finally:
        context.close()


def _write_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
