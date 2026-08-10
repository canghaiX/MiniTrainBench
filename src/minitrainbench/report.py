from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def render_report(paths: list[str]) -> str:
    payloads = [_load(path) for path in paths]
    training = [item for item in payloads if item.get("benchmark") == "training"]
    communication = [
        row
        for item in payloads
        if item.get("benchmark") == "communication"
        for row in item.get("results", [])
    ]
    lines = [
        "## Generated Benchmark Results",
        "",
        "### Training",
        "",
        "| Strategy | GPUs | Precision | Tokens/sec | Step time (ms) | Max memory (MB) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
        lines.append(
            f"| {item['strategy']} | {item['world_size']} | {item['precision']} | "
            f"{item['tokens_per_sec']:.2f} | {item['step_time_ms']:.2f} | "
            f"{item['max_cuda_memory_mb']:.2f} |"
        )
    lines.extend(
        [
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
    return "\n".join(lines) + "\n"


def write_report(paths: list[str], output: str | None) -> str:
    report = render_report(paths)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report)
    return report
