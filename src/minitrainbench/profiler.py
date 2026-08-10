from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .runtime import StepMetrics, Trainer, _write_json


def _set_trainer_defaults(args: Any) -> None:
    args.steps = args.profile_active
    args.warmup_steps = args.profile_warmup
    args.repeat = 1
    args.checkpoint_dir = None
    args.save_every = 0
    args.keep_last = 0
    args.resume = None


def _summarize_steps(metrics: list[StepMetrics]) -> dict[str, dict[str, float]]:
    fields = [
        "data_time_ms",
        "forward_backward_ms",
        "optimizer_step_ms",
        "step_time_ms",
        "tokens_per_sec",
    ]
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(getattr(item, field)) for item in metrics]
        summary[field] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _event_time_us(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, 0.0)
        if value:
            return float(value)
    return 0.0


def _top_events(
    profiler: torch.profiler.profile,
    metric_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        total_us = _event_time_us(event, *metric_names)
        if total_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "calls": int(getattr(event, "count", 0)),
                "time_ms": total_us / 1000,
                "self_cpu_time_ms": _event_time_us(event, "self_cpu_time_total")
                / 1000,
                "self_cuda_time_ms": _event_time_us(
                    event,
                    "self_cuda_time_total",
                    "self_device_time_total",
                )
                / 1000,
            }
        )
    return sorted(rows, key=lambda row: row["time_ms"], reverse=True)[:15]


def _collective_events(profiler: torch.profiler.profile) -> list[dict[str, Any]]:
    keywords = ("nccl", "gloo", "all_reduce", "all_gather", "reduce_scatter", "broadcast")
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        name = event.key
        if not any(keyword in name.lower() for keyword in keywords):
            continue
        total_us = max(
            _event_time_us(event, "self_cuda_time_total", "self_device_time_total"),
            _event_time_us(event, "self_cpu_time_total"),
        )
        rows.append(
            {
                "name": name,
                "calls": int(getattr(event, "count", 0)),
                "time_ms": total_us / 1000,
            }
        )
    return sorted(rows, key=lambda row: row["time_ms"], reverse=True)[:20]


def _gather_rank_summaries(local_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        return [local_summary]
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_summary)
    return [item for item in gathered if item is not None]


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MiniTrainBench Profiler 摘要",
        "",
        f"- Strategy：{payload['strategy']}",
        f"- GPU 数：{payload['world_size']}",
        f"- 精度：{payload['precision']}",
        f"- Trace 目录：`{payload['trace_dir']}`",
        "",
        "## Step 拆分",
        "",
        "| 指标 | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in payload["step_breakdown"].items():
        lines.append(
            f"| {name} | {values['mean']:.2f} | {values['std']:.2f} | "
            f"{values['min']:.2f} | {values['max']:.2f} |"
        )
    lines.extend(["", "## Rank Top Ops", ""])
    for rank_summary in payload["rank_summaries"]:
        lines.extend(
            [
                f"### Rank {rank_summary['rank']}",
                "",
                "| 类型 | Name | Calls | Time (ms) |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in rank_summary["top_cuda_ops"][:8]:
            lines.append(
                f"| CUDA | {row['name']} | {row['calls']} | {row['time_ms']:.2f} |"
            )
        for row in rank_summary["collectives"][:8]:
            lines.append(
                f"| Collective | {row['name']} | {row['calls']} | {row['time_ms']:.2f} |"
            )
        if not rank_summary["top_cuda_ops"] and not rank_summary["collectives"]:
            for row in rank_summary["top_cpu_ops"][:8]:
                lines.append(
                    f"| CPU | {row['name']} | {row['calls']} | {row['time_ms']:.2f} |"
                )
        lines.append("")
    lines.append(
        "原始 Chrome trace 文件通常较大，默认不提交到 Git；请在本地用 "
        "`chrome://tracing` 或 Perfetto 打开 rank trace。"
    )
    return "\n".join(lines) + "\n"


def profile_training(args: Any) -> dict[str, Any] | None:
    _set_trainer_defaults(args)
    trainer = Trainer(args)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if trainer.context.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.reset_peak_memory_stats(trainer.context.device)
        total_steps = args.profile_wait + args.profile_warmup + args.profile_active
        schedule = torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=1,
        )
        measured: list[StepMetrics] = []
        trainer.model.train()
        trainer.context.barrier()
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=args.record_shapes,
            with_stack=args.with_stack,
            profile_memory=True,
        ) as profiler:
            for step_index in range(total_steps):
                metrics = trainer._run_one_step()
                profiler.step()
                if step_index >= args.profile_wait + args.profile_warmup:
                    measured.append(metrics)
        trainer.context.barrier()

        trace_path = trace_dir / f"rank_{trainer.context.rank:05d}.trace.json"
        profiler.export_chrome_trace(str(trace_path))
        local_summary = {
            "rank": trainer.context.rank,
            "trace_file": str(trace_path),
            "top_cpu_ops": _top_events(profiler, ("self_cpu_time_total",)),
            "top_cuda_ops": _top_events(
                profiler,
                ("self_cuda_time_total", "self_device_time_total"),
            ),
            "collectives": _collective_events(profiler),
        }
        (trace_dir / f"rank_{trainer.context.rank:05d}_summary.json").write_text(
            json.dumps(local_summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        rank_summaries = _gather_rank_summaries(local_summary)
        if not trainer.context.is_main:
            return None

        payload = {
            "benchmark": "profile",
            "strategy": trainer.config.strategy,
            "world_size": trainer.context.world_size,
            "precision": trainer.config.precision,
            "device": str(trainer.context.device),
            "backend": trainer.config.backend,
            "trace_dir": str(trace_dir),
            "trace_files": [item["trace_file"] for item in rank_summaries],
            "profile_schedule": {
                "wait": args.profile_wait,
                "warmup": args.profile_warmup,
                "active": args.profile_active,
                "record_shapes": bool(args.record_shapes),
                "with_stack": bool(args.with_stack),
            },
            "gradient_accumulation_steps": trainer.config.grad_accum_steps,
            "gradient_sync_mode": trainer.config.gradient_sync_mode,
            "resolved_gradient_sync_mode": trainer.resolved_gradient_sync_mode,
            "activation_checkpointing": trainer.config.activation_checkpointing,
            "parameters": trainer.parameter_count,
            "model_config": trainer.config.model_dict(),
            "step_breakdown": _summarize_steps(measured),
            "rank_summaries": rank_summaries,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(trainer.context.local_rank)
                if trainer.context.device.type == "cuda"
                else None,
            },
        }
        summary_path = trace_dir / "profile_summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (trace_dir / "profile_summary.md").write_text(_render_markdown(payload))
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        if trainer._owns_context:
            trainer.context.close()
