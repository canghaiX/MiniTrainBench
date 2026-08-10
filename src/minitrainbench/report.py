from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _metric(item: dict[str, Any], name: str) -> float:
    summary = item.get("summary", {})
    if name in summary:
        return float(summary[name]["mean"])
    return float(item[name])


def _repeat_count(item: dict[str, Any]) -> int:
    return int(item.get("repeat_count") or len(item.get("repeats", [])) or 1)


def _format_optional(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.2f}{suffix}"


def render_report(paths: list[str]) -> str:
    payloads = [_load(path) for path in paths]
    training = [item for item in payloads if item.get("benchmark") == "training"]
    communication = [
        row
        for item in payloads
        if item.get("benchmark") == "communication"
        for row in item.get("results", [])
    ]
    by_strategy = {
        strategy: sorted(
            [item for item in training if item["strategy"] == strategy],
            key=lambda row: row["world_size"],
        )
        for strategy in {item["strategy"] for item in training}
    }
    baseline_tokens = {
        strategy: _metric(items[0], "tokens_per_sec")
        for strategy, items in by_strategy.items()
        if items and items[0]["world_size"] == 1
    }
    ddp_memory = {
        item["world_size"]: _metric(item, "max_cuda_memory_mb")
        for item in training
        if item["strategy"] == "ddp"
    }
    ddp_step_time = {
        item["world_size"]: _metric(item, "step_time_ms")
        for item in training
        if item["strategy"] == "ddp"
    }

    lines = [
        "## Generated Benchmark Results",
        "",
        "### Training",
        "",
        "| Strategy | GPUs | Precision | Tokens/sec | Step time (ms) | Max memory (MB) | Scaling efficiency | Memory saving vs DDP | Step delta vs DDP (ms) | Repeats |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
        tokens_per_sec = _metric(item, "tokens_per_sec")
        step_time_ms = _metric(item, "step_time_ms")
        max_memory_mb = _metric(item, "max_cuda_memory_mb")
        baseline = baseline_tokens.get(item["strategy"])
        scaling = (
            tokens_per_sec / (baseline * item["world_size"]) * 100
            if baseline
            else None
        )
        memory_saving = None
        step_delta = None
        if item["strategy"] == "fsdp" and item["world_size"] in ddp_memory:
            memory_saving = (
                (ddp_memory[item["world_size"]] - max_memory_mb)
                / ddp_memory[item["world_size"]]
                * 100
            )
            step_delta = step_time_ms - ddp_step_time[item["world_size"]]
        lines.append(
            f"| {item['strategy']} | {item['world_size']} | {item['precision']} | "
            f"{tokens_per_sec:.2f} | {step_time_ms:.2f} | {max_memory_mb:.2f} | "
            f"{_format_optional(scaling, '%')} | "
            f"{_format_optional(memory_saving, '%')} | "
            f"{_format_optional(step_delta)} | {_repeat_count(item)} |"
        )
    lines.extend(
        [
            "",
            "Scaling efficiency is normalized to each strategy's 1-GPU throughput. "
            "FSDP memory saving and step delta are computed against DDP at the same GPU count.",
            "",
            "### Communication",
            "",
            "| Operation | GPUs | Elements | Latency (ms) | Bandwidth (GB/s) | Status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in communication:
        latency = f"{item['latency_ms']:.3f}" if item["status"] == "ok" else "-"
        bandwidth = f"{item['bandwidth_gbps']:.3f}" if item["status"] == "ok" else "-"
        lines.append(
            f"| {item['operation']} | {item['world_size']} | {item['elements']} | "
            f"{latency} | {bandwidth} | {item['status']} |"
        )
    if communication:
        lines.extend(
            [
                "",
                "Small collective sizes are latency-bound; larger tensors expose bandwidth limits. "
                "Compare these rows with training step time to estimate communication pressure.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_report(paths: list[str], output: str | None) -> str:
    report = render_report(paths)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report)
    return report
