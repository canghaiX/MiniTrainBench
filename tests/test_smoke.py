from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from minitrainbench.model import GPTConfig, MiniGPT


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
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["strategy"] == "ddp"
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
