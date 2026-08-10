from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from minitrainbench.model import GPTConfig, MiniGPT
from minitrainbench.report import render_report


def _run_module(*arguments: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "minitrainbench", *arguments],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )


def test_tiny_gpt_forward_backward() -> None:
    model = MiniGPT(
        GPTConfig(vocab_size=128, seq_length=16, d_model=32, n_heads=4, n_layers=1),
        activation_checkpointing=True,
    )
    input_ids = torch.randint(0, 128, (2, 16))
    _, loss = model(input_ids, input_ids)
    assert loss is not None
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cpu_training_smoke(tmp_path) -> None:
    output = tmp_path / "train.json"
    completed = _run_module(
        "train",
        "--device",
        "cpu",
        "--strategy",
        "ddp",
        "--precision",
        "fp32",
        "--batch-size",
        "1",
        "--seq-length",
        "16",
        "--vocab-size",
        "128",
        "--d-model",
        "32",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
        "--repeat",
        "2",
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["strategy"] == "ddp"
    assert result["repeat_count"] == 2
    assert len(result["repeats"]) == 2
    assert "summary" in result
    assert result["tokens_per_sec"] > 0
    assert "step_time_ms" in completed.stdout


def test_gloo_collectives_smoke(tmp_path) -> None:
    output = tmp_path / "comm.json"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "minitrainbench",
            "comm",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--sizes",
            "32",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
        timeout=180,
    )
    result = json.loads(output.read_text())
    statuses = {row["operation"]: row["status"] for row in result["results"]}
    assert statuses["all_reduce"] == "ok"
    assert statuses["all_gather"] == "ok"
    assert "all_reduce" in completed.stdout


def test_report_renders_infra_metrics(tmp_path) -> None:
    payloads = {
        "ddp_1.json": {
            "benchmark": "training",
            "strategy": "ddp",
            "world_size": 1,
            "precision": "bf16",
            "tokens_per_sec": 100.0,
            "step_time_ms": 10.0,
            "max_cuda_memory_mb": 1000.0,
            "repeat_count": 1,
        },
        "ddp_2.json": {
            "benchmark": "training",
            "strategy": "ddp",
            "world_size": 2,
            "precision": "bf16",
            "tokens_per_sec": 150.0,
            "step_time_ms": 12.0,
            "max_cuda_memory_mb": 1100.0,
            "repeat_count": 1,
        },
        "fsdp_1.json": {
            "benchmark": "training",
            "strategy": "fsdp",
            "world_size": 1,
            "precision": "bf16",
            "tokens_per_sec": 80.0,
            "step_time_ms": 14.0,
            "max_cuda_memory_mb": 900.0,
            "repeat_count": 1,
        },
        "fsdp_2.json": {
            "benchmark": "training",
            "strategy": "fsdp",
            "world_size": 2,
            "precision": "bf16",
            "tokens_per_sec": 120.0,
            "step_time_ms": 16.0,
            "max_cuda_memory_mb": 550.0,
            "repeat_count": 2,
            "summary": {
                "tokens_per_sec": {"mean": 120.0, "std": 2.0, "min": 118.0, "max": 122.0},
                "step_time_ms": {"mean": 16.0, "std": 0.2, "min": 15.8, "max": 16.2},
                "max_cuda_memory_mb": {"mean": 550.0, "std": 0.0, "min": 550.0, "max": 550.0},
            },
        },
        "comm.json": {
            "benchmark": "communication",
            "results": [
                {
                    "operation": "all_reduce",
                    "world_size": 2,
                    "elements": 1024,
                    "latency_ms": 0.1,
                    "bandwidth_gbps": 1.0,
                    "status": "ok",
                }
            ],
        },
    }
    paths = []
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        paths.append(str(path))

    report = render_report(paths)

    assert "Scaling efficiency" in report
    assert "75.00%" in report
    assert "50.00%" in report
    assert "4.00" in report
    assert "Small collective sizes are latency-bound" in report
