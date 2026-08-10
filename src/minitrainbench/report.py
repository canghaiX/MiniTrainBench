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


def _optional_metric(item: dict[str, Any], name: str) -> float | None:
    if name not in item and name not in item.get("summary", {}):
        return None
    return _metric(item, name)


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
        "## 生成的 Benchmark 结果",
        "",
        "### 训练",
        "",
        "| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | "
        "Tokens/sec | Step time (ms) | 最大显存 (MB) | "
        "扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
        tokens_per_sec = _metric(item, "tokens_per_sec")
        step_time_ms = _metric(item, "step_time_ms")
        max_memory_mb = _metric(item, "max_cuda_memory_mb")
        data_time_ms = _optional_metric(item, "data_time_ms")
        forward_backward_ms = _optional_metric(item, "forward_backward_ms")
        optimizer_step_ms = _optional_metric(item, "optimizer_step_ms")
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
            f"{_format_optional(data_time_ms)} | "
            f"{_format_optional(forward_backward_ms)} | "
            f"{_format_optional(optimizer_step_ms)} | "
            f"{tokens_per_sec:.2f} | {step_time_ms:.2f} | {max_memory_mb:.2f} | "
            f"{_format_optional(scaling, '%')} | "
            f"{_format_optional(memory_saving, '%')} | "
            f"{_format_optional(step_delta)} | {_repeat_count(item)} |"
        )
    lines.extend(
        [
            "",
            "扩展效率以同一策略的 1 卡吞吐为基准归一化。"
            "FSDP 显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。",
            "",
            "### Runtime 状态",
            "",
            "| 策略 | GPU 数 | 是否恢复 | Global step | Tokens seen | Checkpoint |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
        runtime = item.get("runtime", {})
        resumed = runtime.get("resume", item.get("resumed", False))
        global_step = runtime.get("global_step", item.get("global_step", "-"))
        tokens_seen = runtime.get("tokens_seen", item.get("tokens_seen", "-"))
        checkpoint = runtime.get(
            "last_checkpoint",
            item.get("checkpoint_dir", "-"),
        ) or "-"
        lines.append(
            f"| {item['strategy']} | {item['world_size']} | "
            f"{'是' if resumed else '否'} | {global_step} | {tokens_seen} | "
            f"{checkpoint} |"
        )
    lines.extend(
        [
            "",
            "### 通信",
            "",
            "| 操作 | GPU 数 | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |",
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
                "小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。"
                "可将这些结果与训练 step time 对比，用于估计通信压力。",
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
