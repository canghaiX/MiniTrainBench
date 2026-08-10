from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from minitrainbench.data import SyntheticTokenIterator
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


def test_synthetic_iterator_is_step_deterministic() -> None:
    iterator = SyntheticTokenIterator(
        vocab_size=64,
        batch_size=2,
        seq_length=8,
        seed=2026,
        rank=0,
    )
    same_step = iterator.batch_for_step(3, torch.device("cpu"))
    same_step_again = iterator.batch_for_step(3, torch.device("cpu"))
    next_step = iterator.batch_for_step(4, torch.device("cpu"))
    other_rank = SyntheticTokenIterator(
        vocab_size=64,
        batch_size=2,
        seq_length=8,
        seed=2026,
        rank=1,
    ).batch_for_step(3, torch.device("cpu"))

    assert torch.equal(same_step, same_step_again)
    assert not torch.equal(same_step, next_step)
    assert not torch.equal(same_step, other_rank)


def test_cpu_checkpoint_resume_and_config_guard(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first_output = tmp_path / "first.json"
    resume_output = tmp_path / "resume.json"
    common_arguments = [
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
        "8",
        "--vocab-size",
        "64",
        "--d-model",
        "16",
        "--n-heads",
        "4",
        "--n-layers",
        "1",
        "--warmup-steps",
        "0",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]

    _run_module(
        *common_arguments,
        "--steps",
        "2",
        "--save-every",
        "1",
        "--output",
        str(first_output),
        timeout=240,
    )
    first_result = json.loads(first_output.read_text())
    assert first_result["global_step"] == 2
    assert first_result["tokens_seen"] == 16
    assert (checkpoint_dir / "step_00000002" / "READY").is_file()
    assert "配置指纹" in (checkpoint_dir / "step_00000002" / "metadata_zh.md").read_text()
    assert (checkpoint_dir / "latest").read_text().strip() == "step_00000002"

    _run_module(
        *common_arguments,
        "--steps",
        "1",
        "--resume",
        "latest",
        "--save-every",
        "1",
        "--output",
        str(resume_output),
        timeout=240,
    )
    resumed_result = json.loads(resume_output.read_text())
    assert resumed_result["runtime"]["resume"] is True
    assert resumed_result["runtime"]["resume_path"].endswith("step_00000002")
    assert resumed_result["global_step"] == 3
    assert resumed_result["tokens_seen"] == 24
    assert resumed_result["runtime"]["last_checkpoint"].endswith("step_00000003")
    assert (checkpoint_dir / "latest").read_text().strip() == "step_00000003"

    bad_environment = os.environ.copy()
    bad_environment["OMP_NUM_THREADS"] = "1"
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "minitrainbench",
            *common_arguments,
            "--d-model",
            "32",
            "--steps",
            "1",
            "--resume",
            str(checkpoint_dir / "step_00000002"),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=bad_environment,
        timeout=240,
    )
    assert bad.returncode != 0
    assert "与当前训练配置不匹配" in bad.stderr


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

    assert "扩展效率" in report
    assert "Runtime 状态" in report
    assert "前反向" in report
    assert "75.00%" in report
    assert "50.00%" in report
    assert "4.00" in report
    assert "小规模 collective 更容易受延迟限制" in report
