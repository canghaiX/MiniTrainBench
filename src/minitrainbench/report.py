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


def _metric_summary(item: dict[str, Any], name: str) -> dict[str, float] | None:
    summary = item.get("summary", {})
    if name not in summary:
        return None
    values = summary[name]
    return {
        "mean": float(values["mean"]),
        "std": float(values["std"]),
        "min": float(values["min"]),
        "max": float(values["max"]),
    }


def _repeat_count(item: dict[str, Any]) -> int:
    return int(item.get("repeat_count") or len(item.get("repeats", [])) or 1)


def _format_optional(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.2f}{suffix}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "是" if value else "否"


def _format_metric(item: dict[str, Any], name: str) -> str:
    summary = _metric_summary(item, name)
    if summary is None:
        if name not in item:
            return "-"
        return f"{float(item[name]):.2f}"
    return f"{summary['mean']:.2f} ± {summary['std']:.2f}"


def _format_runtime_number(value: Any) -> str:
    return "-" if value in (None, "-") else f"{float(value):.6g}"


def _provenance_warnings(payloads: list[dict[str, Any]]) -> list[str]:
    warnings = []
    incomplete = [
        item.get("benchmark", "unknown")
        for item in payloads
        if not item.get("provenance", {}).get("complete", False)
    ]
    if incomplete:
        warnings.append(
            "存在缺少完整 provenance 的输入：" + ", ".join(sorted(set(incomplete)))
        )
    signatures = {
        (
            item.get("provenance", {}).get("git_revision"),
            item.get("provenance", {}).get("image_id"),
            item.get("provenance", {}).get("base_image"),
        )
        for item in payloads
        if item.get("provenance", {}).get("complete", False)
    }
    if len(signatures) > 1:
        warnings.append("输入结果混用了源码 revision、容器 image ID 或 base image。")
    return warnings


def _environment_rows(payloads: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    rows = {
        (
            item.get("environment", {}).get("torch"),
            item.get("environment", {}).get("cuda"),
            item.get("environment", {}).get("cudnn"),
            item.get("environment", {}).get("nccl"),
            item.get("environment", {}).get("driver"),
            item.get("environment", {}).get("gpu"),
            item.get("provenance", {}).get("git_revision"),
            item.get("provenance", {}).get("image_id"),
            item.get("provenance", {}).get("base_image"),
            item.get("provenance", {}).get("complete", False),
        )
        for item in payloads
    }
    return sorted(rows, key=lambda row: tuple(str(value) for value in row))


def render_report(paths: list[str]) -> str:
    payloads = [_load(path) for path in paths]
    training = [item for item in payloads if item.get("benchmark") == "training"]
    communication = [
        row
        for item in payloads
        if item.get("benchmark") == "communication"
        for row in item.get("results", [])
    ]
    tensor_parallel = [
        item for item in payloads if item.get("benchmark") == "tensor_parallel"
    ]
    tensor_parallel_mlp = [
        item for item in payloads if item.get("benchmark") == "tensor_parallel_mlp"
    ]
    sequence_parallel = [
        item for item in payloads if item.get("benchmark") == "sequence_parallel"
    ]
    moe_routing = [item for item in payloads if item.get("benchmark") == "moe_routing"]
    fault_tolerance = [
        item for item in payloads if item.get("benchmark") == "fault_tolerance"
    ]
    doctors = [item for item in payloads if item.get("benchmark") == "doctor"]
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

    provenance_warnings = _provenance_warnings(payloads)
    environment_rows = _environment_rows(payloads)

    lines = [
        "## 生成的 Benchmark 结果",
        "",
    ]
    if provenance_warnings:
        lines.extend(
            [
                "> **实验环境警告：** " + " ".join(provenance_warnings),
                "",
            ]
        )
    lines.extend(
        [
            "### 实验环境",
            "",
            (
                "| PyTorch | CUDA | cuDNN | NCCL | Driver | GPU | Git revision | "
                "Image ID | Base image | Provenance |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in environment_rows:
        torch_version, cuda, cudnn, nccl, driver, gpu, revision, image_id, base, complete = row
        lines.append(
            f"| {torch_version or '-'} | {cuda or '-'} | {cudnn or '-'} | "
            f"{nccl or '-'} | {driver or '-'} | {gpu or '-'} | "
            f"{str(revision)[:12] if revision else '-'} | "
            f"{str(image_id)[:19] if image_id else '-'} | {base or '-'} | "
            f"{'完整' if complete else '不完整'} |"
        )
    lines.extend(
        [
            "",
        "### 训练",
        "",
        (
            "| 策略 | GPU 数 | 精度 | Data (ms) | 前反向 (ms) | 优化器 (ms) | "
            "Tokens/sec | Step time (ms) | 最大显存 (MB) | "
            "扩展效率 | 相对 DDP 显存节省 | 相对 DDP step 差值 (ms) | Repeats |"
        ),
        (
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
        ]
    )
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
        if item["strategy"] != "ddp" and item["world_size"] in ddp_memory:
            memory_saving = (
                (ddp_memory[item["world_size"]] - max_memory_mb)
                / ddp_memory[item["world_size"]]
                * 100
            )
            step_delta = step_time_ms - ddp_step_time[item["world_size"]]
        lines.append(
            f"| {item['strategy']} | {item['world_size']} | {item['precision']} | "
            f"{_format_metric(item, 'data_time_ms')} | "
            f"{_format_metric(item, 'forward_backward_ms')} | "
            f"{_format_metric(item, 'optimizer_step_ms')} | "
            f"{_format_metric(item, 'tokens_per_sec')} | "
            f"{_format_metric(item, 'step_time_ms')} | "
            f"{_format_metric(item, 'max_cuda_memory_mb')} | "
            f"{_format_optional(scaling, '%')} | "
            f"{_format_optional(memory_saving, '%')} | "
            f"{_format_optional(step_delta)} | {_repeat_count(item)} |"
        )
    lines.extend(
        [
            "",
            (
                "扩展效率以同一策略的 1 卡吞吐为基准归一化。"
                "非 DDP 策略的显存节省和 step 差值均与相同 GPU 数下的 DDP 对比计算。"
            ),
            "",
            "### Runtime 状态",
            "",
            (
                "| 策略 | GPU 数 | Strategy impl | 是否恢复 | Global step | Tokens seen | "
                "Trial protocol | 请求同步 | 实际同步 | 同步 micro-batch/step | 精确恢复 | "
                "Latest | Keep last | Ready 数 | Resume path | Last checkpoint |"
            ),
            (
                "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | "
                "--- | ---: | ---: | --- | --- |"
            ),
        ]
    )
    for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
        runtime = item.get("runtime", {})
        resumed = runtime.get("resume", item.get("resumed", False))
        global_step = runtime.get("global_step", item.get("global_step", "-"))
        tokens_seen = runtime.get("tokens_seen", item.get("tokens_seen", "-"))
        last_checkpoint = runtime.get(
            "last_checkpoint",
            item.get("checkpoint_dir", "-"),
        ) or "-"
        latest_checkpoint = runtime.get("latest_checkpoint", "-") or "-"
        keep_last = runtime.get("keep_last", "-")
        ready_checkpoints = runtime.get("ready_checkpoints", "-")
        strategy_impl = runtime.get("strategy_impl", "-")
        resume_path = runtime.get("resume_path", "-") or "-"
        trial_protocol = runtime.get(
            "trial_protocol",
            item.get("trial_protocol", "-"),
        )
        gradient_sync_mode = runtime.get(
            "gradient_sync_mode",
            item.get("gradient_sync_mode", "-"),
        )
        resolved_gradient_sync_mode = runtime.get(
            "resolved_gradient_sync_mode",
            item.get("resolved_gradient_sync_mode", "-"),
        )
        synchronized_microbatches = runtime.get(
            "synchronized_microbatches_per_step",
            item.get("synchronized_microbatches_per_step", "-"),
        )
        resume_deterministic = runtime.get("resume_deterministic")
        lines.append(
            f"| {item['strategy']} | {item['world_size']} | {strategy_impl} | "
            f"{'是' if resumed else '否'} | {global_step} | {tokens_seen} | "
            f"{trial_protocol} | {gradient_sync_mode} | {resolved_gradient_sync_mode} | "
            f"{synchronized_microbatches} | {_format_optional_bool(resume_deterministic)} | "
            f"{latest_checkpoint} | {keep_last} | {ready_checkpoints} | "
            f"{resume_path} | {last_checkpoint} |"
        )
    if training:
        lines.extend(
            [
                "",
                "#### 稳定性指标",
                "",
                (
                    "| 策略 | GPU 数 | LR scheduler | 当前 LR | Grad norm mean | "
                    "Grad norm max | 裁剪阈值 | 裁剪步数 | 非有限值策略 |"
                ),
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in sorted(training, key=lambda row: (row["strategy"], row["world_size"])):
            runtime = item.get("runtime", {})
            lr_scheduler = runtime.get(
                "lr_scheduler", item.get("config", {}).get("lr_scheduler", "-")
            )
            learning_rate = runtime.get("learning_rate", item.get("learning_rate", "-"))
            grad_norm = runtime.get("grad_norm", item.get("grad_norm", "-"))
            grad_norm_max = item.get("grad_norm_max", grad_norm)
            max_grad_norm = runtime.get(
                "max_grad_norm", item.get("config", {}).get("max_grad_norm", "-")
            )
            clipped_steps = runtime.get("clipped_steps", item.get("clipped_steps", "-"))
            nonfinite_policy = runtime.get(
                "nonfinite_policy", item.get("nonfinite_policy", "-")
            )
            lines.append(
                f"| {item['strategy']} | {item['world_size']} | {lr_scheduler} | "
                f"{_format_runtime_number(learning_rate)} | "
                f"{_format_runtime_number(grad_norm)} | "
                f"{_format_runtime_number(grad_norm_max)} | "
                f"{_format_runtime_number(max_grad_norm)} | "
                f"{_format_runtime_number(clipped_steps)} | "
                f"{nonfinite_policy} |"
            )
    lines.extend(
        [
            "",
            "### 通信",
            "",
            "| 操作 | GPU 数 | Split | 元素数 | 延迟 (ms) | 带宽 (GB/s) | 状态 |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in communication:
        latency = f"{item['latency_ms']:.3f}" if item["status"] == "ok" else "-"
        bandwidth = f"{item['bandwidth_gbps']:.3f}" if item["status"] == "ok" else "-"
        split_mode = item.get("split_mode", "-") or "-"
        lines.append(
            f"| {item['operation']} | {item['world_size']} | {split_mode} | "
            f"{item['elements']} | {latency} | {bandwidth} | {item['status']} |"
        )
    if communication:
        lines.extend(
            [
                "",
                (
                    "小规模 collective 更容易受延迟限制；较大 tensor 更能暴露带宽上限。"
                    "all-to-all 对应 MoE expert parallel 的 token dispatch/combine，"
                    "可将这些结果与训练 step time 对比，用于估计稀疏模型通信压力。"
                ),
            ]
        )
    if tensor_parallel:
        lines.extend(
            [
                "",
                "### Tensor Parallel 正确性",
                "",
                (
                    "| TP degree | Device | In | Out | Forward max error | "
                    "Grad max error | 状态 |"
                ),
                "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in sorted(tensor_parallel, key=lambda row: row["tp_degree"]):
            lines.append(
                f"| {item['tp_degree']} | {item['device']} | "
                f"{item['in_features']} | {item['out_features']} | "
                f"{item['forward_max_error']:.6g} | "
                f"{item['grad_max_error']:.6g} | {item['status']} |"
            )
        lines.extend(
            [
                "",
                (
                    "toy TP 校验把 ColumnParallelLinear 和 RowParallelLinear "
                    "与单卡 reference 对齐，用于说明 Megatron-style tensor parallel "
                    "的切分语义和梯度聚合路径。"
                ),
            ]
        )
    if tensor_parallel_mlp or sequence_parallel:
        lines.extend(
            [
                "",
                "### Megatron-style Toy Runtime 正确性",
                "",
                (
                    "| 类型 | TP degree | Device | Shape | Forward max error | "
                    "Grad max error | Collectives | 通信量估算 (bytes) | 状态 |"
                ),
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in sorted(tensor_parallel_mlp, key=lambda row: row["tp_degree"]):
            shape = (
                f"{item['in_features']}->{item['hidden_features']}->"
                f"{item['out_features']}"
            )
            lines.append(
                f"| TP MLP | {item['tp_degree']} | {item['device']} | {shape} | "
                f"{item['forward_max_error']:.6g} | {item['grad_max_error']:.6g} | "
                f"{item['collective_count']} | {item['communication_bytes']} | "
                f"{item['status']} |"
            )
        for item in sorted(sequence_parallel, key=lambda row: row["tp_degree"]):
            shape = f"seq={item['seq_length']}, hidden={item['hidden_size']}"
            lines.append(
                f"| Sequence Parallel | {item['tp_degree']} | {item['device']} | "
                f"{shape} | {item['forward_max_error']:.6g} | "
                f"{item['grad_max_error']:.6g} | {item['collective_count']} | "
                f"{item['communication_bytes']} | {item['status']} |"
            )
        lines.extend(
            [
                "",
                (
                    "TP MLP 展示 ColumnParallel + RowParallel 如何组成一段可反传的 "
                    "Megatron-style MLP；Sequence Parallel 展示 LayerNorm/Dropout "
                    "在 sequence shard 上的 correctness 边界。"
                ),
            ]
        )
    if moe_routing:
        lines.extend(
            [
                "",
                "### MoE Routing / Expert Parallel",
                "",
                (
                    "| GPU 数 | Experts | Tokens/rank | Capacity | Drop tokens | "
                    "Imbalance | Load-balance loss | Dispatch (ms) | Combine (ms) | 状态 |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in moe_routing:
            lines.append(
                f"| {item.get('world_size', '-')} | {item.get('num_experts', '-')} | "
                f"{item.get('tokens_per_rank', '-')} | "
                f"{item.get('capacity_per_expert', '-')} | "
                f"{item.get('tokens_dropped', '-')} | "
                f"{float(item.get('load_imbalance_ratio', 0.0)):.2f} | "
                f"{float(item.get('load_balance_loss', 0.0)):.4f} | "
                f"{float(item.get('dispatch_time_ms', 0.0)):.3f} | "
                f"{float(item.get('combine_time_ms', 0.0)):.3f} | "
                f"{item.get('status', '-')} |"
            )
        lines.extend(
            [
                "",
                (
                    "toy MoE routing 记录 top-1 router、capacity、overflow、负载不均衡和 "
                    "dispatch/combine 通信，用于解释 expert parallel 的系统瓶颈。"
                ),
            ]
        )
    if fault_tolerance:
        lines.extend(
            [
                "",
                "### Failure Handling",
                "",
                (
                    "| 故障类型 | 检测方式 | 自动恢复 | 恢复模式 | Checkpoint 未变 | "
                    "恢复 checkpoint | Global step | Tokens seen | 状态 |"
                ),
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in fault_tolerance:
            for row in item.get("failure_handling", []):
                recovered = row.get("recovered_checkpoint") or "-"
                lines.append(
                    f"| {row.get('failure_type', '-')} | {row.get('detection', '-')} | "
                    f"{_format_optional_bool(row.get('auto_recovered'))} | "
                    f"{row.get('recovery_mode', '-')} | "
                    f"{_format_optional_bool(row.get('checkpoint_unchanged'))} | "
                    f"{recovered} | {row.get('global_step', '-')} | "
                    f"{row.get('tokens_seen', '-')} | {row.get('status', '-')} |"
                )
        lines.extend(
            [
                "",
                (
                    "该表覆盖最小故障模型：精确 resume、半成品 checkpoint 跳过、配置不匹配拒绝、"
                    "NaN、rank crash 和通信 timeout 的检测边界。"
                ),
            ]
        )
    if doctors:
        lines.extend(
            [
                "",
                "### Doctor 环境诊断",
                "",
                "| GPU 数 | CUDA | NCCL | Connectivity | Diagnostics |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for item in doctors:
            diagnostics = "; ".join(
                f"{row['level']}:{row['check']}" for row in item.get("diagnostics", [])
            )
            lines.append(
                f"| {item.get('gpu_count', '-')} | "
                f"{_format_optional_bool(item.get('cuda_available'))} | "
                f"{item.get('nccl_version') or '-'} | "
                f"{item.get('connectivity', {}).get('status', '-')} | "
                f"{diagnostics or '-'} |"
            )
    return "\n".join(lines) + "\n"


def write_report(paths: list[str], output: str | None) -> str:
    report = render_report(paths)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report)
    return report
