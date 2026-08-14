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
DEPENDENCY_PATTERNS = (
    "no module named",
    "no such file or directory",
    "failed to compile the c++ dataset helper",
    "please install te",
    "must install apex",
)


def _summary(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("至少需要一个样本")
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


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
    if "timed out" in text or "watchdog timeout" in text:
        return "failed", "通信或进程超时"
    if any(pattern in text for pattern in DEPENDENCY_PATTERNS):
        return "failed", "外部运行环境缺少依赖或编译工具"
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


def parse_device_memory_samples(path: str | None) -> dict[str, Any] | None:
    """解析 nvidia-smi 设备显存采样，显式区分基线、峰值和增量。"""
    if not path or not Path(path).is_file():
        return None
    devices: dict[int, dict[str, list[float]]] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or fields[0] not in {"baseline", "sample"}:
            continue
        try:
            index = int(fields[2])
            memory_mb = float(fields[3])
        except ValueError:
            continue
        devices.setdefault(index, {"baseline": [], "sample": []})[fields[0]].append(
            memory_mb
        )
    rows = []
    for index, samples in sorted(devices.items()):
        baseline_values = samples["baseline"]
        measured_values = samples["sample"]
        if not baseline_values or not measured_values:
            continue
        baseline = baseline_values[0]
        peak = max(measured_values)
        rows.append(
            {
                "device_index": index,
                "baseline_device_memory_mb": baseline,
                "peak_device_memory_mb": peak,
                "peak_device_memory_delta_mb": max(0.0, peak - baseline),
            }
        )
    if not rows:
        return None
    return {
        "source": "nvidia_smi_device_used_memory",
        "devices": rows,
        "peak_device_memory_mb": max(row["peak_device_memory_mb"] for row in rows),
        "peak_device_memory_delta_mb": max(
            row["peak_device_memory_delta_mb"] for row in rows
        ),
    }


def build_megatron_trial_record(
    *,
    config: dict[str, Any],
    log_text: str,
    returncode: int,
    command: str,
    environment: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    memory_samples_path: str | None = None,
) -> dict[str, Any]:
    """构造单次 Megatron trial 的公开、可审计记录。"""
    status, reason = classify_failure(returncode, log_text)
    is_timeout = (
        returncode == 124
        or "timed out" in log_text.lower()
        or "watchdog timeout" in log_text.lower()
    )
    if is_timeout:
        status = "timeout"
    parsed = parse_megatron_log(log_text)
    measured_iters = int(config.get("measured_iters", 0))
    all_step_samples = parsed.get("step_time_ms_samples", [])
    measured_steps = all_step_samples[-measured_iters:] if measured_iters > 0 else []
    metrics: dict[str, Any] = {
        "step_sample_count": len(measured_steps),
        "step_time_ms": None,
        "step_time_ms_summary": None,
        "tokens_per_sec": None,
        "tokens_per_sec_source": None,
        "parameters": parsed.get("parameters"),
    }
    if measured_steps:
        step_summary = _summary(measured_steps)
        metrics["step_time_ms"] = step_summary["mean"]
        metrics["step_time_ms_summary"] = step_summary
        metrics["tokens_per_sec"] = (
            int(config["global_batch_size"])
            * int(config["seq_length"])
            / (step_summary["mean"] / 1000)
        )
        metrics["tokens_per_sec_source"] = (
            "derived_global_batch_seq_length_over_mean_step_time"
        )
    memory = parse_device_memory_samples(memory_samples_path)
    if memory:
        metrics.update(
            {
                "peak_device_memory_mb": memory["peak_device_memory_mb"],
                "peak_device_memory_delta_mb": memory[
                    "peak_device_memory_delta_mb"
                ],
                "device_memory": memory["devices"],
                "memory_source": memory["source"],
            }
        )
    if status == "success" and len(measured_steps) != measured_iters:
        status = "failed_parse"
        reason = (
            f"期望 {measured_iters} 个测量 step，日志中仅找到 "
            f"{len(measured_steps)} 个"
        )
    dp = int(config["dp"])
    pp = int(config["pp"])
    micro_batches = int(config["global_batch_size"]) // (
        int(config["micro_batch_size"]) * dp
    )
    bubble_proxy = (pp - 1) / (micro_batches + pp - 1)
    record = {
        "benchmark": "megatron",
        **config,
        "status": status,
        "failure_reason": reason or None,
        "returncode": returncode,
        "metrics": metrics,
        "runtime": {
            "micro_batches_per_iteration": micro_batches,
            "pipeline_bubble_proxy": bubble_proxy,
            "pipeline_bubble_proxy_source": "theoretical_fill_drain_approximation",
            "pipeline_idle_observed": "not_determined_without_trace",
        },
        "command": command.strip(),
        "log_sha256": hashlib.sha256(log_text.encode()).hexdigest(),
        "environment": environment,
        "provenance": provenance,
    }
    return record


def aggregate_megatron_trials(
    trials: Iterable[dict[str, Any]], *, expected_repeats: int
) -> dict[str, Any]:
    """聚合同一 TP/PP/DP 配置的独立进程 trial。"""
    rows = list(trials)
    if not rows:
        raise ValueError("Megatron 聚合至少需要一个 trial")
    first = rows[0]
    topology = (
        first.get("tp"),
        first.get("pp"),
        first.get("dp"),
        first.get("world_size"),
    )
    if any(
        (row.get("tp"), row.get("pp"), row.get("dp"), row.get("world_size"))
        != topology
        for row in rows[1:]
    ):
        raise ValueError("不能聚合 TP/PP/DP/world size 不一致的 Megatron trial")
    statuses = [str(row.get("status")) for row in rows]
    successful = [row for row in rows if row.get("status") == "success"]
    complete = len(rows) == expected_repeats and len(successful) == expected_repeats
    if complete:
        status = "success"
        reason = None
    elif successful:
        status = "partial"
        reason = f"{len(successful)}/{expected_repeats} 个 trial 成功"
    elif "oom" in statuses:
        status = "oom"
        reason = "全部 trial 未成功，至少一个 trial OOM"
    elif "failed_parse" in statuses:
        status = "failed_parse"
        reason = "全部 trial 未成功，至少一个 trial 无法解析完整指标"
    elif "timeout" in statuses:
        status = "timeout"
        reason = "全部 trial 未成功，至少一个 trial 超时"
    else:
        status = "failed"
        reason = "全部 trial 均失败"
    summary: dict[str, Any] = {}
    metric_names = (
        "tokens_per_sec",
        "step_time_ms",
        "peak_device_memory_mb",
        "peak_device_memory_delta_mb",
    )
    for name in metric_names:
        values = [
            row.get("metrics", {}).get(name)
            for row in successful
            if row.get("metrics", {}).get(name) is not None
        ]
        if values:
            summary[name] = _summary(values)
    return {
        "benchmark": "megatron",
        "name": first.get("name"),
        "tp": first.get("tp"),
        "pp": first.get("pp"),
        "dp": first.get("dp"),
        "world_size": first.get("world_size"),
        "precision": first.get("precision", "bf16"),
        "model_config": first.get("model_config"),
        "batch_config": first.get("batch_config"),
        "runtime": first.get("runtime"),
        "evidence_mode": first.get("evidence_mode", "formal"),
        "performance_valid": all(
            row.get("performance_valid", True) for row in rows
        ),
        "performance_invalid_reasons": sorted(
            {
                reason
                for row in rows
                for reason in row.get("performance_invalid_reasons", [])
            }
        ),
        "status": status,
        "failure_reason": reason,
        "repeat_count": len(rows),
        "expected_repeat_count": expected_repeats,
        "trial_protocol": "independent_process_restart",
        "summary": summary,
        "trials": [
            {
                "trial_index": row.get("trial_index"),
                "status": row.get("status"),
                "failure_reason": row.get("failure_reason"),
                "metrics": row.get("metrics"),
                "log_sha256": row.get("log_sha256"),
                "command": row.get("command"),
            }
            for row in rows
        ],
        "environment": first.get("environment"),
        "provenance": first.get("provenance"),
        "megatron": first.get("megatron"),
    }


def public_result_violations(payload: dict[str, Any]) -> list[str]:
    """检查公开结果中不应出现的凭据、代理和宿主机私有路径。"""
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    checks = {
        "ngc_api_key": "NGC API Key",
        "https_proxy": "代理变量",
        "http_proxy": "代理变量",
        "_json_key": "NGC 登录用户名",
        "/data/external/": "外部源码绝对路径",
        "/root/.docker/": "Docker 凭据路径",
    }
    violations = {label for pattern, label in checks.items() if pattern in serialized}
    if re.search(r"https?://[^/\s:@]+:[^@\s]+@", serialized):
        violations.add("URL 内嵌凭据")
    return sorted(violations)


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
        (
            "| 配置 | 环境 | TP | PP | DP | Repeat | 状态 | 性能可比 | Tokens/sec | Step (ms) | "
            "设备峰值显存 (MB) | 理论 bubble proxy | 失败原因 |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | "
            "---: | ---: | --- |"
        ),
    ]

    if any(not row.get("performance_valid", True) for row in rows):
        reasons = sorted(
            {
                reason
                for row in rows
                for reason in row.get("performance_invalid_reasons", [])
            }
        )
        lines[4:4] = [
            "**警告：本报告包含兼容性 smoke，性能指标不可横向比较。**",
            "原因：" + ", ".join(reasons),
            "",
        ]

    def formatted_summary(row: dict[str, Any], name: str) -> str:
        if not row.get("performance_valid", True):
            return "-"
        summary = row.get("summary", {}).get(name)
        if summary:
            return f"{summary['mean']:.2f} +/- {summary['std']:.2f}"
        value = row.get("metrics", {}).get(name)
        return f"{float(value):.2f}" if value is not None else "-"

    for row in rows:
        lines.append(
            f"| {row.get('name', '-')} | "
            f"{row.get('megatron', {}).get('environment_profile', '-')} | "
            f"{row.get('tp', '-')} | {row.get('pp', '-')} | "
            f"{row.get('dp', '-')} | {row.get('repeat_count', 1)} | "
            f"{row.get('status', '-')} | "
            f"{'yes' if row.get('performance_valid', True) else 'no'} | "
            f"{formatted_summary(row, 'tokens_per_sec')} | "
            f"{formatted_summary(row, 'step_time_ms')} | "
            f"{formatted_summary(row, 'peak_device_memory_mb')} | "
            f"{row.get('runtime', {}).get('pipeline_bubble_proxy', '-')} | "
            f"{row.get('failure_reason') or '-'} |"
        )
    lines.extend(
        [
            "",
            "`Tokens/sec` 由 global batch、sequence length 与测量 step 均值推导；设备显存来自",
            "独占 GPU 条件下的 `nvidia-smi` 设备级采样，不等同于 PyTorch allocator 指标。",
            "Pipeline bubble 仅给出 fill-drain 理论 proxy；没有 trace 时不声称观察到 idle。",
            "兼容性 smoke 的原始指标保留在 JSON 供审计，但 Markdown 不展示为性能结论。",
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
    megatron_record.add_argument("--command", required=True)
    megatron_record.add_argument("--environment", default=None)
    megatron_record.add_argument("--provenance", default=None)
    megatron_record.add_argument("--memory-samples", default=None)
    megatron_record.add_argument("--output", required=True)

    megatron_aggregate = subparsers.add_parser("megatron-aggregate")
    megatron_aggregate.add_argument("--input", nargs="+", required=True)
    megatron_aggregate.add_argument("--expected-repeats", required=True, type=int)
    megatron_aggregate.add_argument("--output", required=True)

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
        record = build_megatron_trial_record(
            config=config,
            log_text=log_text,
            returncode=args.returncode,
            command=args.command,
            environment=_read_json(args.environment) if args.environment else None,
            provenance=_read_json(args.provenance) if args.provenance else None,
            memory_samples_path=args.memory_samples,
        )
        violations = public_result_violations(record)
        if violations:
            raise ValueError("公开结果审计失败：" + "、".join(violations))
        _write(args.output, record)
    elif args.action == "megatron-aggregate":
        record = aggregate_megatron_trials(
            (_read_json(path) for path in args.input),
            expected_repeats=args.expected_repeats,
        )
        violations = public_result_violations(record)
        if violations:
            raise ValueError("公开结果审计失败：" + "、".join(violations))
        _write(args.output, record)
    elif args.action == "megatron-report":
        _write(args.output, render_megatron_report(_read_json(path) for path in args.input))
    elif args.action == "manifest":
        _write(args.output, build_evidence_manifest(args.input))


if __name__ == "__main__":
    main()
