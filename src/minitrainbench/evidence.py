"""外部 benchmark 证据的归一化、失败分类和 Markdown 渲染。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

OOM_PATTERNS = ("out of memory", "cuda out of memory", "cublas_status_alloc_failed")


def validate_megatron_config(
    *,
    world_size: int,
    tensor_parallel: int,
    pipeline_parallel: int,
    micro_batch_size: int,
    global_batch_size: int,
) -> int:
    """校验 TP/PP/DP 与 batch 约束，并返回 data parallel degree。"""
    model_parallel = tensor_parallel * pipeline_parallel
    if min(world_size, tensor_parallel, pipeline_parallel, micro_batch_size) < 1:
        raise ValueError("world size、TP、PP 和 micro batch 必须为正数")
    if world_size % model_parallel != 0:
        raise ValueError("world size 必须能被 TP*PP 整除")
    data_parallel = world_size // model_parallel
    if global_batch_size % (micro_batch_size * data_parallel) != 0:
        raise ValueError("global batch 必须能被 micro batch*DP 整除")
    return data_parallel


def classify_failure(returncode: int, output: str) -> tuple[str, str]:
    text = output.lower()
    if any(pattern in text for pattern in OOM_PATTERNS):
        return "oom", "CUDA 内存不足"
    if "timed out" in text or "timeout" in text or "watchdog" in text:
        return "failed", "通信或进程超时"
    if returncode == 0:
        return "success", ""
    return "failed", "外部训练命令返回非零状态"


def training_record(
    *,
    benchmark_id: str,
    config: dict[str, Any],
    command: list[str] | str,
    output: str,
    returncode: int,
    result_path: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, reason = classify_failure(returncode, output)
    record: dict[str, Any] = {
        "benchmark": "memory_pressure",
        "benchmark_id": benchmark_id,
        "status": status,
        "failure_reason": reason or None,
        "failure_type": None,
        "config": config,
        "command": command,
        "returncode": returncode,
        "strategy": config.get("strategy"),
        "world_size": config.get("world_size"),
        "precision": config.get("precision"),
        "environment": (evidence or {}).get("environment"),
        "provenance": (evidence or {}).get("provenance"),
    }
    if status == "oom":
        record["failure_type"] = "cuda_oom"
    elif returncode == 124 or "timeout" in output.lower() or "watchdog" in output.lower():
        record["failure_type"] = "timeout"
    elif returncode != 0:
        record["failure_type"] = "process_error"
    if result_path and Path(result_path).is_file():
        try:
            result = json.loads(Path(result_path).read_text())
            record.update(
                {
                    "strategy": result.get("strategy"),
                    "world_size": result.get("world_size"),
                    "precision": result.get("precision"),
                    "parameters": result.get("parameters"),
                    "tokens_per_sec": result.get("tokens_per_sec"),
                    "step_time_ms": result.get("step_time_ms"),
                    "max_cuda_memory_mb": result.get("max_cuda_memory_mb"),
                    "summary": result.get("summary"),
                    "repeat_count": result.get("repeat_count"),
                    "model_config": result.get("model_config"),
                    "environment": result.get("environment"),
                    "provenance": result.get("provenance"),
                    "source_result": result_path,
                }
            )
        except (OSError, json.JSONDecodeError):
            record["status"] = "failed_parse"
            record["failure_reason"] = "训练结果 JSON 无法解析"
            record["failure_type"] = "result_parse_error"
    elif status == "success":
        record["status"] = "failed_parse"
        record["failure_reason"] = "命令成功，但没有生成训练结果 JSON"
        record["failure_type"] = "missing_result"
    return record


def parse_megatron_log(text: str) -> dict[str, Any]:
    """解析常见 Megatron 日志字段，缺失字段保持 None。"""
    patterns = {
        "step_time_ms": (
            r"(?:elapsed time per iteration|iteration time)\s*"
            r"\(?ms\)?\s*[:=]\s*([0-9.]+)"
        ),
        "tokens_per_sec": r"tokens(?:/sec| per second)\s*[:=]\s*([0-9.]+)",
        "max_memory_mb": (
            r"(?:max memory(?: allocated)?|max allocated|memory allocated|allocated memory)"
            r"\s*[:=]\s*"
            r"([0-9.]+)\s*(?:MiB|MB)?"
        ),
        "parameters": r"(?:number of parameters|parameters)\s*[:=]\s*([0-9]+)",
    }
    result: dict[str, Any] = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        result[name] = float(matches[-1]) if matches else None
        if name in {"step_time_ms", "tokens_per_sec", "max_memory_mb"}:
            result[f"{name}_samples"] = [float(value) for value in matches]
    if result["parameters"] is not None:
        result["parameters"] = int(result["parameters"])
    return result


def render_memory_pressure(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    tier_parameters = {
        str(row.get("config", {}).get("tier")): row.get("parameters")
        for row in rows
        if row.get("parameters") is not None
    }

    def metric(row: dict[str, Any], name: str, fallback: Any = None) -> str:
        value = row.get(name, fallback)
        if value is None:
            return "-"
        if name == "parameters":
            return f"{int(value) / 1_000_000:.1f}M"
        return f"{float(value):.2f}"

    lines = [
        "# MiniTrainBench 显存压力矩阵",
        "",
        "每个档位独立初始化；OOM 和解析失败也是实验结果，不会被静默丢弃。",
        "",
        (
            "| 档位 | 策略 | GPU 数 | 模型配置 | Batch/Seq | AC | 目标参数量 | 状态 | "
            "Tokens/sec | Step (ms) | 峰值显存 (MB) | 原因 |"
        ),
        "| --- | --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        config = row.get("config", {})
        target_parameters = tier_parameters.get(str(config.get("tier")))
        model = (
            f"L{config.get('n_layers', '-')}/H{config.get('d_model', '-')}/"
            f"A{config.get('n_heads', '-')}"
        )
        lines.append(
            f"| {config.get('tier', '-')} | {row.get('strategy', '-')} | "
            f"{row.get('world_size', '-')} | {model} | "
            f"{config.get('batch_size', '-')}/{config.get('seq_length', '-')} | "
            f"{config.get('activation_checkpointing', '-')} | "
            f"{metric(row, 'parameters', target_parameters)} | "
            f"{row.get('status', '-')} | {metric(row, 'tokens_per_sec')} | "
            f"{metric(row, 'step_time_ms')} | {metric(row, 'max_cuda_memory_mb')} | "
            f"{row.get('failure_reason') or '-'} |"
        )
    lines.extend(
        [
            "",
            "解释边界：small 模型主要观察框架开销；medium/large 才更接近参数、梯度和优化器",
            "状态分片的实际收益。未采集的指标显示为 `-`，不根据模型规模估算显存。",
        ]
    )
    lines.extend(["", "## 自动分析", ""])
    by_tier: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        tier = str(row.get("config", {}).get("tier", "unknown"))
        by_tier.setdefault(tier, {})[str(row.get("strategy"))] = row
    for tier in ("small", "medium", "large", "stress"):
        strategies = by_tier.get(tier, {})
        ddp = strategies.get("ddp")
        fsdp = strategies.get("fsdp")
        if not ddp or not fsdp:
            continue
        if ddp.get("status") == "oom" and fsdp.get("status") == "success":
            lines.append(
                f"- `{tier}`：DDP OOM，FSDP 成功，峰值显存 "
                f"{metric(fsdp, 'max_cuda_memory_mb')} MB。"
            )
            continue
        if ddp.get("status") != "success" or fsdp.get("status") != "success":
            continue
        ddp_memory = float(ddp["max_cuda_memory_mb"])
        fsdp_memory = float(fsdp["max_cuda_memory_mb"])
        ddp_throughput = float(ddp["tokens_per_sec"])
        fsdp_throughput = float(fsdp["tokens_per_sec"])
        memory_saving = (1 - fsdp_memory / ddp_memory) * 100
        throughput_delta = (fsdp_throughput / ddp_throughput - 1) * 100
        lines.append(
            f"- `{tier}`：FSDP 相对 DDP 节省 {memory_saving:.1f}% 峰值显存，"
            f"吞吐变化 {throughput_delta:+.1f}%。"
        )
    lines.extend(["", "## 完整启动命令", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('benchmark_id', '-')}",
                "",
                "```bash",
                str(row.get("command", "-")).strip(),
                "```",
                "",
            ]
        )
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def render_megatron_report(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    lines = [
        "# Megatron-LM 8 卡 Smoke / TP-PP-DP 矩阵",
        "",
        (
            "Megatron 源码由用户通过 `MEGATRON_DIR` 提供；本目录只保存命令、版本、"
            "日志解析结果和失败原因。"
        ),
        "",
        "| 配置 | TP | PP | DP | 状态 | Tokens/sec | Step (ms) | Peak memory (MB) | 失败原因 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            f"| {row.get('name', '-')} | {row.get('tp', '-')} | {row.get('pp', '-')} | "
            f"{row.get('dp', '-')} | {row.get('status', '-')} | "
            f"{metrics.get('tokens_per_sec') or '-'} | {metrics.get('step_time_ms') or '-'} | "
            f"{metrics.get('max_memory_mb') or '-'} | {row.get('failure_reason') or '-'} |"
        )
    lines.extend(
        [
            "",
            "没有真实数据集和完整生产配置；本实验只用于验证并行组、启动参数和性能指标链路。",
            "解析缺失字段显示为 `-`，不把日志无法证明的指标写入结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_evidence_manifest(paths: Iterable[str]) -> dict[str, Any]:
    entries = []
    for raw_path in sorted(paths):
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        provenance = payload.get("provenance", {})
        entries.append(
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "benchmark": payload.get("benchmark"),
                "status": payload.get("status", "success"),
                "world_size": payload.get("world_size"),
                "strategy": payload.get("strategy"),
                "command": provenance.get("command"),
                "git_revision": provenance.get("git_revision"),
                "image_ref": provenance.get("image_ref"),
                "image_id": provenance.get("image_id"),
                "base_image": provenance.get("base_image"),
                "provenance_complete": provenance.get("complete", False),
                "provenance_missing_fields": provenance.get("missing_fields", []),
            }
        )
    complete = bool(entries) and all(row["provenance_complete"] for row in entries)
    return {
        "benchmark": "evidence_manifest",
        "complete": complete,
        "entry_count": len(entries),
        "entries": entries,
    }


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write(path: str, payload: dict[str, Any] | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        destination.write_text(payload)
    else:
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="归一化外部 benchmark 证据")
    subparsers = parser.add_subparsers(dest="action", required=True)

    memory_record = subparsers.add_parser("memory-record")
    memory_record.add_argument("--benchmark-id", required=True)
    memory_record.add_argument("--config-json", required=True)
    memory_record.add_argument("--command", dest="benchmark_command", required=True)
    memory_record.add_argument("--log", required=True)
    memory_record.add_argument("--returncode", required=True, type=int)
    memory_record.add_argument("--result", default=None)
    memory_record.add_argument("--evidence", default=None)
    memory_record.add_argument("--output", required=True)

    memory_report = subparsers.add_parser("memory-report")
    memory_report.add_argument("--input", nargs="+", required=True)
    memory_report.add_argument("--output", required=True)

    megatron_record = subparsers.add_parser("megatron-record")
    megatron_record.add_argument("--config-json", required=True)
    megatron_record.add_argument("--log", required=True)
    megatron_record.add_argument("--returncode", required=True, type=int)
    megatron_record.add_argument("--output", required=True)

    megatron_report = subparsers.add_parser("megatron-report")
    megatron_report.add_argument("--input", nargs="+", required=True)
    megatron_report.add_argument("--output", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--input", nargs="+", required=True)
    manifest.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.action == "memory-record":
        config = json.loads(args.config_json)
        record = training_record(
            benchmark_id=args.benchmark_id,
            config=config,
            command=args.benchmark_command,
            output=Path(args.log).read_text(errors="replace"),
            returncode=args.returncode,
            result_path=args.result,
            evidence=_read_json(args.evidence) if args.evidence else None,
        )
        _write(args.output, record)
    elif args.action == "memory-report":
        _write(args.output, render_memory_pressure(_read_json(path) for path in args.input))
    elif args.action == "megatron-record":
        config = json.loads(args.config_json)
        log_text = Path(args.log).read_text(errors="replace")
        status, reason = classify_failure(args.returncode, log_text)
        metrics = parse_megatron_log(log_text)
        measured_iters = int(config.get("measured_iters", 0))
        step_samples = metrics.get("step_time_ms_samples", [])
        if measured_iters > 0 and step_samples:
            measured_steps = step_samples[-measured_iters:]
            metrics["step_time_ms"] = statistics.fmean(measured_steps)
            metrics["step_time_ms_min"] = min(measured_steps)
            metrics["step_time_ms_max"] = max(measured_steps)
        if metrics.get("tokens_per_sec") is None and metrics.get("step_time_ms"):
            metrics["tokens_per_sec"] = (
                int(config["global_batch_size"])
                * int(config["seq_length"])
                / (float(metrics["step_time_ms"]) / 1000)
            )
            metrics["tokens_per_sec_source"] = "derived_from_step_time"
        if status == "success" and not any(
            metrics.get(name) is not None
            for name in ("tokens_per_sec", "step_time_ms", "max_memory_mb")
        ):
            status = "failed_parse"
            reason = "命令成功，但日志中未找到性能指标"
        record = {
            "benchmark": "megatron",
            **config,
            "status": status,
            "failure_reason": reason or None,
            "returncode": args.returncode,
            "metrics": metrics,
            "log_file": args.log,
        }
        _write(args.output, record)
    elif args.action == "megatron-report":
        _write(args.output, render_megatron_report(_read_json(path) for path in args.input))
    elif args.action == "manifest":
        _write(args.output, build_evidence_manifest(args.input))


if __name__ == "__main__":
    main()
