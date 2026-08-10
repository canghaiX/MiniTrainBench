from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from minitrainbench.checkpoint import CheckpointManager
from minitrainbench.communication import _all_to_all_splits
from minitrainbench.data import SyntheticTokenIterator
from minitrainbench.deepspeed_benchmark import build_deepspeed_config
from minitrainbench.distributed import DistributedContext
from minitrainbench.model import GPTConfig, MiniGPT
from minitrainbench.report import render_report
from minitrainbench.runtime import TrainingConfig
from minitrainbench.strategy import create_strategy, registered_strategies
from minitrainbench.verification import _validate_pair


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
    assert result["trial_protocol"] == "independent_reinitialize"
    assert len(result["repeats"]) == 2
    assert [repeat["global_step"] for repeat in result["repeats"]] == [1, 1]
    assert [repeat["tokens_seen"] for repeat in result["repeats"]] == [16, 16]
    assert "summary" in result
    assert result["tokens_per_sec"] > 0
    assert "step_time_ms" in completed.stdout


def test_strategy_registry_creates_supported_strategies() -> None:
    assert registered_strategies() == ("ddp", "fsdp")
    assert create_strategy("ddp").name() == "DDPStrategy"
    assert create_strategy("ddp").requires_process_group() is False
    assert create_strategy("fsdp").name() == "FSDPStrategy"
    assert create_strategy("fsdp").requires_process_group() is True
    with pytest.raises(ValueError, match="不支持的训练策略"):
        create_strategy("unknown")


def test_deepspeed_config_builder_zero_stages() -> None:
    args = SimpleNamespace(
        batch_size=2,
        grad_accum_steps=4,
        precision="bf16",
        zero_stage=2,
    )
    zero2 = build_deepspeed_config(args, world_size=8)
    assert zero2["train_batch_size"] == 64
    assert zero2["bf16"]["enabled"] is True
    assert zero2["zero_optimization"]["stage"] == 2
    assert "stage3_prefetch_bucket_size" not in zero2["zero_optimization"]

    args.zero_stage = 3
    args.precision = "fp32"
    zero3 = build_deepspeed_config(args, world_size=2)
    assert zero3["train_batch_size"] == 16
    assert zero3["bf16"]["enabled"] is False
    assert zero3["zero_optimization"]["stage"] == 3
    assert "stage3_prefetch_bucket_size" in zero3["zero_optimization"]


def test_all_to_all_split_generation() -> None:
    world_size = 4
    base = 8
    equal_inputs, equal_outputs = _all_to_all_splits(base, world_size, rank=2, mode="equal")
    assert equal_inputs == [base, base, base, base]
    assert equal_outputs == [base, base, base, base]

    all_inputs = [
        _all_to_all_splits(base, world_size, rank=rank, mode="uneven")[0]
        for rank in range(world_size)
    ]
    for rank in range(world_size):
        input_splits, output_splits = _all_to_all_splits(
            base,
            world_size,
            rank=rank,
            mode="uneven",
        )
        assert input_splits == all_inputs[rank]
        assert output_splits == [all_inputs[source][rank] for source in range(world_size)]
        assert sum(input_splits) == sum(output_splits)


def test_gradient_sync_policy_and_no_sync_context() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.entries = 0

        @contextmanager
        def no_sync(self):
            self.entries += 1
            yield

    model = FakeModel()
    ddp = create_strategy("ddp")
    fsdp = create_strategy("fsdp")

    assert ddp.resolve_gradient_sync_mode("auto") == "last"
    assert fsdp.resolve_gradient_sync_mode("auto") == "every"
    assert ddp.resolve_gradient_sync_mode("every") == "every"
    assert fsdp.resolve_gradient_sync_mode("last") == "last"

    with ddp.gradient_sync_context(model, sync_gradients=False):
        pass
    with ddp.gradient_sync_context(model, sync_gradients=True):
        pass
    assert model.entries == 1


def test_cpu_gradient_sync_result(tmp_path) -> None:
    output = tmp_path / "gradient-sync.json"
    _run_module(
        "train",
        "--device",
        "cpu",
        "--strategy",
        "ddp",
        "--precision",
        "fp32",
        "--gradient-sync-mode",
        "auto",
        "--grad-accum-steps",
        "2",
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
        "--steps",
        "1",
        "--warmup-steps",
        "0",
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["gradient_sync_mode"] == "auto"
    assert result["resolved_gradient_sync_mode"] == "last"
    assert result["synchronized_microbatches_per_step"] == 1
    assert result["runtime"]["resume_deterministic"] is None


def test_cpu_profiler_smoke(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    _run_module(
        "profile",
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
        "--profile-wait",
        "0",
        "--profile-warmup",
        "0",
        "--profile-active",
        "1",
        "--trace-dir",
        str(trace_dir),
        timeout=240,
    )
    summary = json.loads((trace_dir / "profile_summary.json").read_text())
    assert summary["benchmark"] == "profile"
    assert summary["world_size"] == 1
    assert summary["step_breakdown"]["step_time_ms"]["mean"] > 0
    assert (trace_dir / "rank_00000.trace.json").is_file()
    assert (trace_dir / "profile_summary.md").is_file()


def test_cpu_doctor_smoke(tmp_path) -> None:
    output = tmp_path / "doctor.json"
    _run_module(
        "doctor",
        "--device",
        "cpu",
        "--backend",
        "gloo",
        "--skip-connectivity",
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["benchmark"] == "doctor"
    assert "diagnostics" in result
    assert result["connectivity"]["status"] == "skipped"


def test_cpu_moe_routing_smoke(tmp_path) -> None:
    output = tmp_path / "moe.json"
    _run_module(
        "moe",
        "route",
        "--device",
        "cpu",
        "--backend",
        "gloo",
        "--tokens-per-rank",
        "8",
        "--hidden-size",
        "8",
        "--num-experts",
        "4",
        "--capacity-factor",
        "1.25",
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["benchmark"] == "moe_routing"
    assert result["status"] == "ok"
    assert sum(result["expert_token_counts"]) + result["tokens_dropped"] == 8
    assert result["top_k"] == 1


def test_cpu_fault_tolerance_smoke(tmp_path) -> None:
    output = tmp_path / "fault.json"
    _run_module(
        "fault",
        "smoke",
        "--device",
        "cpu",
        "--backend",
        "gloo",
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
        "--dropout",
        "0.2",
        "--continuous-steps",
        "2",
        "--interrupted-steps",
        "1",
        "--resume-steps",
        "1",
        "--output",
        str(output),
        timeout=360,
    )
    result = json.loads(output.read_text())
    assert result["benchmark"] == "fault_tolerance"
    assert result["verification"]["exact_match"] is True
    failure_types = {row["failure_type"] for row in result["failure_handling"]}
    assert {"checkpoint_resume_exact", "config_mismatch", "half_checkpoint"} <= failure_types


def test_repeat_conflicts_with_checkpoint_args(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "minitrainbench",
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
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--repeat",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        timeout=60,
    )
    assert failed.returncode != 0
    assert "独立 benchmark trial" in failed.stderr


def test_gloo_ddp_gradient_accumulation_auto_smoke(tmp_path) -> None:
    output = tmp_path / "gradient-sync-ddp.json"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "minitrainbench",
            "train",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--strategy",
            "ddp",
            "--precision",
            "fp32",
            "--gradient-sync-mode",
            "auto",
            "--grad-accum-steps",
            "2",
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
            "--steps",
            "1",
            "--warmup-steps",
            "0",
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
    assert result["world_size"] == 2
    assert result["resolved_gradient_sync_mode"] == "last"
    assert result["synchronized_microbatches_per_step"] == 1


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
        "--keep-last",
        "1",
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
    assert first_result["runtime"]["strategy_impl"] == "DDPStrategy"
    assert first_result["runtime"]["keep_last"] == 1
    assert first_result["runtime"]["ready_checkpoints"] == 1
    assert first_result["runtime"]["latest_checkpoint"] == "step_00000002"
    assert (checkpoint_dir / "step_00000002" / "READY").is_file()
    assert not (checkpoint_dir / "step_00000001").exists()
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
    assert resumed_result["runtime"]["ready_checkpoints"] == 1
    assert (checkpoint_dir / "latest").read_text().strip() == "step_00000003"
    assert not (checkpoint_dir / "step_00000002").exists()

    bad_checkpoint = checkpoint_dir / "step_99999999"
    bad_checkpoint.mkdir()
    (checkpoint_dir / "latest").write_text("step_99999999\n")
    manager = CheckpointManager(
        str(checkpoint_dir),
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized_here=False,
        ),
    )
    assert manager.find_latest() == checkpoint_dir / "step_00000003"
    inspected = manager.inspect("step_00000003")
    assert inspected["ready"] is True
    assert inspected["step"] == 3

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
            str(checkpoint_dir / "step_00000003"),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=bad_environment,
        timeout=240,
    )
    assert bad.returncode != 0
    assert "与当前训练配置不匹配" in bad.stderr


def test_checkpoint_verify_exact_resume_and_legacy_diagnostic(tmp_path) -> None:
    continuous_dir = tmp_path / "continuous"
    interrupted_dir = tmp_path / "interrupted"
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
        "--dropout",
        "0.2",
        "--warmup-steps",
        "0",
        "--keep-last",
        "0",
    ]
    _run_module(
        *common_arguments,
        "--steps",
        "3",
        "--checkpoint-dir",
        str(continuous_dir),
        "--save-every",
        "3",
        timeout=240,
    )
    _run_module(
        *common_arguments,
        "--steps",
        "2",
        "--checkpoint-dir",
        str(interrupted_dir),
        "--save-every",
        "2",
        timeout=240,
    )
    _run_module(
        *common_arguments,
        "--steps",
        "1",
        "--checkpoint-dir",
        str(interrupted_dir),
        "--resume",
        "latest",
        "--save-every",
        "1",
        timeout=240,
    )
    verification_output = tmp_path / "verification.json"
    _run_module(
        "checkpoint",
        "verify",
        "--device",
        "cpu",
        "--backend",
        "gloo",
        "--left",
        str(continuous_dir / "step_00000003"),
        "--right",
        str(interrupted_dir / "step_00000003"),
        "--output",
        str(verification_output),
        timeout=240,
    )
    verification = json.loads(verification_output.read_text())
    assert verification["exact_match"] is True
    assert verification["matches"] == {
        "model": True,
        "optimizer": True,
        "train_state": True,
        "rng": True,
    }

    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        initialized_here=False,
    )
    manager = CheckpointManager(root=None, context=context)
    with pytest.raises(ValueError, match="READY"):
        _validate_pair(
            manager,
            str(continuous_dir / "step_00000003"),
            str(tmp_path / "not-ready"),
            context,
        )
    wrong_world_size = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        initialized_here=False,
    )
    with pytest.raises(ValueError, match="world size"):
        _validate_pair(
            CheckpointManager(root=None, context=wrong_world_size),
            str(continuous_dir / "step_00000003"),
            str(interrupted_dir / "step_00000003"),
            wrong_world_size,
        )
    mismatched = tmp_path / "mismatched"
    shutil.copytree(continuous_dir / "step_00000003", mismatched)
    metadata_path = mismatched / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["strategy"] = "fsdp"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="元数据不兼容"):
        _validate_pair(
            manager,
            str(continuous_dir / "step_00000003"),
            str(mismatched),
            context,
        )

    legacy_rng = interrupted_dir / "step_00000003" / "rng_state_rank_00000.pt"
    legacy_rng.unlink()
    legacy_metadata_path = interrupted_dir / "step_00000003" / "metadata.json"
    legacy_metadata = json.loads(legacy_metadata_path.read_text())
    legacy_metadata["format_version"] = 1
    legacy_metadata.pop("rng_state_version", None)
    legacy_config = dict(legacy_metadata["config"])
    legacy_config["gradient_sync_mode"] = "auto"
    legacy_metadata["config_fingerprint"] = TrainingConfig.from_dict(
        legacy_config
    ).legacy_fingerprint()
    legacy_metadata["config"].pop("gradient_sync_mode")
    legacy_metadata_path.write_text(json.dumps(legacy_metadata))
    legacy_output = tmp_path / "legacy.json"
    _run_module(
        *common_arguments,
        "--steps",
        "1",
        "--checkpoint-dir",
        str(interrupted_dir),
        "--resume",
        "step_00000003",
        "--output",
        str(legacy_output),
        timeout=240,
    )
    legacy_result = json.loads(legacy_output.read_text())
    assert legacy_result["runtime"]["resume_deterministic"] is False
    assert (
        legacy_result["runtime"]["resume_determinism_reason"]
        == "checkpoint_missing_rng_state"
    )
    assert legacy_result["resolved_gradient_sync_mode"] == "every"


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
    all_to_all_rows = [
        row for row in result["results"] if row["operation"] == "all_to_all"
    ]
    assert {row["split_mode"] for row in all_to_all_rows} == {"equal", "uneven"}
    assert {row["status"] for row in all_to_all_rows} <= {"ok", "skipped"}
    assert "all_reduce" in completed.stdout


def test_tensor_parallel_cpu_check(tmp_path) -> None:
    output = tmp_path / "tp.json"
    mlp_output = tmp_path / "tp_mlp.json"
    sequence_output = tmp_path / "tp_sequence.json"
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
            "tp",
            "check",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--batch-size",
            "1",
            "--seq-length",
            "2",
            "--in-features",
            "8",
            "--out-features",
            "8",
            "--atol",
            "1e-3",
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
    assert result["benchmark"] == "tensor_parallel"
    assert result["status"] == "ok"
    assert result["tp_degree"] == 2
    assert result["forward_max_error"] <= 1e-3
    assert result["grad_max_error"] <= 1e-3
    assert "Column" not in completed.stderr

    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "minitrainbench",
            "tp",
            "mlp",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--batch-size",
            "1",
            "--seq-length",
            "2",
            "--in-features",
            "8",
            "--hidden-features",
            "16",
            "--out-features",
            "8",
            "--atol",
            "1e-3",
            "--output",
            str(mlp_output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
        timeout=180,
    )
    mlp_result = json.loads(mlp_output.read_text())
    assert mlp_result["benchmark"] == "tensor_parallel_mlp"
    assert mlp_result["status"] == "ok"
    assert mlp_result["collective_count"] == 2

    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "minitrainbench",
            "tp",
            "sequence",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--batch-size",
            "1",
            "--seq-length",
            "2",
            "--hidden-size",
            "8",
            "--dropout",
            "0.1",
            "--atol",
            "1e-3",
            "--output",
            str(sequence_output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
        timeout=180,
    )
    sequence_result = json.loads(sequence_output.read_text())
    assert sequence_result["benchmark"] == "sequence_parallel"
    assert sequence_result["status"] == "ok"
    assert sequence_result["collective_count"] == 3


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
            "runtime": {
                "strategy_impl": "FSDPStrategy",
                "gradient_sync_mode": "auto",
                "resolved_gradient_sync_mode": "every",
                "synchronized_microbatches_per_step": 2,
                "resume": True,
                "resume_path": "results/runtime_resume/fsdp_2proc_ckpt/step_00000002",
                "resume_deterministic": True,
                "latest_checkpoint": "step_00000003",
                "last_checkpoint": "results/runtime_resume/fsdp_2proc_ckpt/step_00000003",
                "keep_last": 1,
                "ready_checkpoints": 1,
                "global_step": 3,
                "tokens_seen": 1536,
            },
        },
        "zero2_1.json": {
            "benchmark": "training",
            "strategy": "deepspeed_zero2",
            "world_size": 1,
            "precision": "bf16",
            "tokens_per_sec": 100.0,
            "step_time_ms": 20.0,
            "max_cuda_memory_mb": 700.0,
            "repeat_count": 1,
            "trial_protocol": "single_run",
            "runtime": {
                "strategy_impl": "DeepSpeedZeRO2",
                "zero_stage": 2,
                "trial_protocol": "single_run",
            },
        },
        "zero2_2.json": {
            "benchmark": "training",
            "strategy": "deepspeed_zero2",
            "world_size": 2,
            "precision": "bf16",
            "tokens_per_sec": 180.0,
            "step_time_ms": 14.0,
            "max_cuda_memory_mb": 500.0,
            "repeat_count": 2,
            "trial_protocol": "independent_reinitialize",
            "summary": {
                "tokens_per_sec": {
                    "mean": 180.0,
                    "std": 5.0,
                    "min": 175.0,
                    "max": 185.0,
                },
                "step_time_ms": {"mean": 14.0, "std": 0.4, "min": 13.6, "max": 14.4},
                "max_cuda_memory_mb": {
                    "mean": 500.0,
                    "std": 1.0,
                    "min": 499.0,
                    "max": 501.0,
                },
            },
            "runtime": {
                "strategy_impl": "DeepSpeedZeRO2",
                "zero_stage": 2,
                "trial_protocol": "independent_reinitialize",
                "resume": False,
                "keep_last": 0,
                "ready_checkpoints": 0,
                "global_step": 25,
                "tokens_seen": 4096,
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
                },
                {
                    "operation": "all_to_all",
                    "world_size": 2,
                    "split_mode": "uneven",
                    "elements": 1024,
                    "latency_ms": 0.2,
                    "bandwidth_gbps": 2.0,
                    "status": "ok",
                }
            ],
        },
        "tp.json": {
            "benchmark": "tensor_parallel",
            "status": "ok",
            "tp_degree": 2,
            "device": "cpu",
            "in_features": 8,
            "out_features": 8,
            "forward_max_error": 0.0,
            "grad_max_error": 0.0,
        },
        "tp_mlp.json": {
            "benchmark": "tensor_parallel_mlp",
            "status": "ok",
            "tp_degree": 2,
            "device": "cpu",
            "in_features": 8,
            "hidden_features": 16,
            "out_features": 8,
            "forward_max_error": 0.0,
            "grad_max_error": 0.0,
            "collective_count": 2,
            "communication_bytes": 128,
        },
        "sequence.json": {
            "benchmark": "sequence_parallel",
            "status": "ok",
            "tp_degree": 2,
            "device": "cpu",
            "seq_length": 8,
            "hidden_size": 16,
            "forward_max_error": 0.0,
            "grad_max_error": 0.0,
            "collective_count": 3,
            "communication_bytes": 256,
        },
        "fault.json": {
            "benchmark": "fault_tolerance",
            "failure_handling": [
                {
                    "failure_type": "checkpoint_resume_exact",
                    "detection": "checkpoint verify",
                    "auto_recovered": True,
                    "recovered_checkpoint": "results/fault/resumed/step_00000002",
                    "global_step": 2,
                    "tokens_seen": 16,
                    "status": "ok",
                },
                {
                    "failure_type": "half_checkpoint",
                    "detection": "latest/READY 扫描",
                    "auto_recovered": True,
                    "recovered_checkpoint": "step_00000002",
                    "status": "ok",
                },
            ],
        },
        "doctor.json": {
            "benchmark": "doctor",
            "gpu_count": 0,
            "cuda_available": False,
            "nccl_version": None,
            "connectivity": {"status": "skipped"},
            "diagnostics": [{"level": "info", "check": "summary"}],
        },
        "moe.json": {
            "benchmark": "moe_routing",
            "status": "ok",
            "world_size": 2,
            "num_experts": 4,
            "tokens_per_rank": 64,
            "capacity_per_expert": 24,
            "tokens_dropped": 1,
            "load_imbalance_ratio": 1.2,
            "load_balance_loss": 0.01,
            "dispatch_time_ms": 0.2,
            "combine_time_ms": 0.3,
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
    assert "Strategy impl" in report
    assert "FSDPStrategy" in report
    assert "请求同步" in report
    assert "实际同步" in report
    assert "精确恢复" in report
    assert "auto" in report
    assert "every" in report
    assert "step_00000003" in report
    assert "前反向" in report
    assert "120.00 ± 2.00" in report
    assert "DeepSpeedZeRO2" in report
    assert "independent_reinitialize" in report
    assert "75.00%" in report
    assert "50.00%" in report
    assert "54.55%" in report
    assert "4.00" in report
    assert "小规模 collective 更容易受延迟限制" in report
    assert "all-to-all 对应 MoE expert parallel" in report
    assert "Tensor Parallel 正确性" in report
    assert "Megatron-style Toy Runtime 正确性" in report
    assert "MoE Routing / Expert Parallel" in report
    assert "Failure Handling" in report
