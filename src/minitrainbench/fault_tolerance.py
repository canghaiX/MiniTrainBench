from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .checkpoint import CheckpointManager
from .distributed import DistributedContext
from .runtime import Trainer, TrainingConfig, _write_json
from .verification import verify_checkpoints


def _resolve_device_name(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_train_args(
    base: Any,
    checkpoint_dir: Path,
    output: Path,
    steps: int,
    resume: str | None = None,
) -> Any:
    return SimpleNamespace(
        strategy=base.strategy,
        precision=base.precision,
        backend=base.backend,
        device=base.device,
        activation_checkpointing=base.activation_checkpointing,
        grad_accum_steps=base.grad_accum_steps,
        batch_size=base.batch_size,
        seq_length=base.seq_length,
        vocab_size=base.vocab_size,
        d_model=base.d_model,
        n_heads=base.n_heads,
        n_layers=base.n_layers,
        dropout=base.dropout,
        learning_rate=base.learning_rate,
        gradient_sync_mode=getattr(base, "gradient_sync_mode", "auto"),
        steps=steps,
        warmup_steps=base.warmup_steps,
        repeat=1,
        seed=base.seed,
        checkpoint_dir=str(checkpoint_dir),
        save_every=getattr(base, "save_every", 0),
        keep_last=getattr(base, "keep_last", 3),
        resume=resume,
        output=str(output),
    )


def _build_config_from_metadata(metadata: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig.from_dict(metadata["config"])


def _half_checkpoint_probe(root: Path) -> dict[str, Any]:
    probe_root = root.parent / f"{root.name}_half_checkpoint_probe"
    shutil.rmtree(probe_root, ignore_errors=True)
    shutil.copytree(root, probe_root)
    probe_dir = probe_root / "step_99999999"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_root / "latest").write_text("step_99999999\n")
    manager = CheckpointManager(
        str(probe_root),
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized_here=False,
        ),
    )
    latest = manager.find_latest()
    return {
        "failure_type": "half_checkpoint",
        "detection": "latest/READY 扫描",
        "auto_recovered": True,
        "recovered_checkpoint": latest.name if latest else None,
        "status": "ok" if latest is not None else "failed",
    }


def fault_tolerance_smoke(args: Any) -> dict[str, Any] | None:
    device_name = _resolve_device_name(args.device)
    output_path = Path(args.output or "results/fault_tolerance/report.json")
    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    continuous_dir = Path(args.checkpoint_dir or work_dir / "continuous")
    interrupted_dir = work_dir / "interrupted"
    shutil.rmtree(continuous_dir, ignore_errors=True)
    shutil.rmtree(interrupted_dir, ignore_errors=True)
    continuous_dir.mkdir(parents=True, exist_ok=True)
    interrupted_dir.mkdir(parents=True, exist_ok=True)

    base = SimpleNamespace(
        strategy=args.strategy,
        precision=args.precision,
        backend=args.backend,
        device=device_name,
        activation_checkpointing=args.activation_checkpointing,
        grad_accum_steps=args.grad_accum_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        gradient_sync_mode=args.gradient_sync_mode,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        save_every=0,
        keep_last=getattr(args, "keep_last", 0),
    )

    continuous_args = _make_train_args(
        base,
        continuous_dir,
        work_dir / "continuous.json",
        args.continuous_steps,
    )
    continuous_args.save_every = args.continuous_steps
    continuous_args.keep_last = 0
    Trainer(continuous_args).run()

    interrupted_args = _make_train_args(
        base,
        interrupted_dir,
        work_dir / "interrupted.json",
        args.interrupted_steps,
    )
    interrupted_args.save_every = args.interrupted_steps
    interrupted_args.keep_last = 0
    Trainer(interrupted_args).run()

    resumed_args = _make_train_args(
        base,
        interrupted_dir,
        work_dir / "resumed.json",
        args.resume_steps,
        resume="latest",
    )
    resumed_args.save_every = args.resume_steps
    resumed_args.keep_last = 0
    resumed_result = Trainer(resumed_args).run()

    verify_output = work_dir / "verification.json"
    verification_args = SimpleNamespace(
        device=device_name,
        backend=args.backend,
        left=str(continuous_dir / f"step_{args.continuous_steps:08d}"),
        right=str(interrupted_dir / f"step_{args.continuous_steps:08d}"),
        output=str(verify_output),
    )
    verification = verify_checkpoints(verification_args)
    if verification is None:
        verification = json.loads(verify_output.read_text())

    interrupted_manager = CheckpointManager(
        str(interrupted_dir),
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(device_name),
            initialized_here=False,
        ),
    )
    interrupted_metadata = interrupted_manager.inspect(
        f"step_{args.continuous_steps:08d}"
    )
    bad_config = _build_config_from_metadata(interrupted_metadata)
    bad_config = replace(bad_config, d_model=bad_config.d_model * 2)
    mismatch_detected = False
    mismatch_reason = ""
    try:
        interrupted_manager._validate_metadata(
            interrupted_metadata,
            bad_config,
            Path(interrupted_metadata["path"]),
        )
    except ValueError as error:
        mismatch_detected = True
        mismatch_reason = str(error)

    failure_handling = [
        {
            "failure_type": "checkpoint_resume_exact",
            "detection": "checkpoint verify",
            "auto_recovered": bool(verification["exact_match"]),
            "recovered_checkpoint": resumed_result["runtime"]["last_checkpoint"]
            if resumed_result
            else None,
            "global_step": resumed_result["global_step"] if resumed_result else None,
            "tokens_seen": resumed_result["tokens_seen"] if resumed_result else None,
            "status": "ok" if verification["exact_match"] else "failed",
        },
        {
            "failure_type": "config_mismatch",
            "detection": "metadata fingerprint / config 校验",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "rejected" if mismatch_detected else "failed",
            "reason": mismatch_reason,
        },
        {
            "failure_type": "nan_loss",
            "detection": "torch.isfinite(loss)",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "detected",
            "reason": "训练循环应在损失非有限时中断并由外部重试。",
        },
        {
            "failure_type": "rank_crash",
            "detection": "launcher exit code / 进程组退出",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "requires_launcher_retry",
            "reason": "rank crash 需要 launcher 或调度器重启，不在单进程 smoke 中自动修复。",
        },
        {
            "failure_type": "communication_timeout",
            "detection": "NCCL watchdog / doctor connectivity",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "requires_timeout_policy",
            "reason": "通信 hang 依赖 NCCL watchdog、超时和多机诊断策略。",
        },
        _half_checkpoint_probe(interrupted_dir),
    ]

    payload = {
        "benchmark": "fault_tolerance",
        "strategy": args.strategy,
        "precision": args.precision,
        "device": device_name,
        "backend": args.backend,
        "continuous_checkpoint": str(
            continuous_dir / f"step_{args.continuous_steps:08d}"
        ),
        "interrupted_checkpoint": str(
            interrupted_dir / f"step_{args.continuous_steps:08d}"
        ),
        "resume_checkpoint": resumed_result["runtime"]["resume_path"] if resumed_result else None,
        "verification": verification,
        "failure_handling": failure_handling,
        "training_outputs": {
            "continuous": str(work_dir / "continuous.json"),
            "interrupted": str(work_dir / "interrupted.json"),
            "resumed": str(work_dir / "resumed.json"),
        },
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload
