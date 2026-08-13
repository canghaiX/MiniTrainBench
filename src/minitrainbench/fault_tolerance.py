from __future__ import annotations

import json
import os
import shutil
import signal
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from .checkpoint import CheckpointManager
from .distributed import DistributedContext, setup_distributed
from .runtime import (
    NonFiniteTrainingError,
    Trainer,
    TrainingConfig,
    _write_json,
)
from .verification import verify_checkpoints


def crash_worker(args: Any) -> None:
    """在指定 global step 杀死一个 worker，用于验证 launcher 与恢复边界。"""

    def inject(trainer: Trainer) -> None:
        if args.crash_rank >= trainer.context.world_size:
            raise ValueError(
                f"crash-rank={args.crash_rank} 超过 world size={trainer.context.world_size}"
            )
        if (
            trainer.context.rank == args.crash_rank
            and trainer.state.global_step == args.crash_at_step
        ):
            os.kill(os.getpid(), signal.SIGKILL)

    Trainer(args, before_step_hook=inject).run()


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
        lr_scheduler=base.lr_scheduler,
        lr_warmup_steps=base.lr_warmup_steps,
        lr_decay_steps=base.lr_decay_steps,
        min_learning_rate=base.min_learning_rate,
        max_grad_norm=base.max_grad_norm,
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
        inject_nonfinite_step=None,
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
    context = setup_distributed(
        args.backend,
        device_name,
        requires_process_group=True,
    )
    output_path = Path(args.output or "results/fault_tolerance/report.json")
    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    continuous_dir = Path(args.checkpoint_dir or work_dir / "continuous")
    interrupted_dir = work_dir / "interrupted"
    nan_dir = work_dir / "nonfinite"
    if context.is_main:
        shutil.rmtree(continuous_dir, ignore_errors=True)
        shutil.rmtree(interrupted_dir, ignore_errors=True)
        shutil.rmtree(nan_dir, ignore_errors=True)
        continuous_dir.mkdir(parents=True, exist_ok=True)
        interrupted_dir.mkdir(parents=True, exist_ok=True)
        nan_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()

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
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        min_learning_rate=args.min_learning_rate,
        max_grad_norm=args.max_grad_norm,
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
    continuous_final_step = args.warmup_steps + args.continuous_steps
    interrupted_final_step = args.warmup_steps + args.interrupted_steps
    continuous_args.save_every = continuous_final_step
    continuous_args.keep_last = 0
    Trainer(continuous_args, context=context).run()

    interrupted_args = _make_train_args(
        base,
        interrupted_dir,
        work_dir / "interrupted.json",
        args.interrupted_steps,
    )
    interrupted_args.save_every = interrupted_final_step
    interrupted_args.keep_last = 0
    Trainer(interrupted_args, context=context).run()

    resumed_args = _make_train_args(
        base,
        interrupted_dir,
        work_dir / "resumed.json",
        args.resume_steps,
        resume="latest",
    )
    resumed_args.save_every = continuous_final_step
    resumed_args.keep_last = 0
    resumed_result = Trainer(resumed_args, context=context).run()

    verify_output = work_dir / "verification.json"
    verification_args = SimpleNamespace(
        device=device_name,
        backend=args.backend,
        left=str(continuous_dir / f"step_{continuous_final_step:08d}"),
        right=str(interrupted_dir / f"step_{continuous_final_step:08d}"),
        output=str(verify_output),
    )
    verification = verify_checkpoints(verification_args)
    if verification is None:
        verification = json.loads(verify_output.read_text())

    nan_base = SimpleNamespace(**vars(base))
    nan_base.warmup_steps = 0
    nan_seed_args = _make_train_args(
        nan_base,
        nan_dir,
        work_dir / "nonfinite_seed.json",
        1,
    )
    nan_seed_args.save_every = 1
    nan_seed_args.keep_last = 0
    Trainer(nan_seed_args, context=context).run()
    nan_resume_args = _make_train_args(
        nan_base,
        nan_dir,
        work_dir / "nonfinite_resume.json",
        1,
        resume="latest",
    )
    nan_resume_args.save_every = 1
    nan_resume_args.keep_last = 0
    nan_resume_args.inject_nonfinite_step = 1
    nan_trainer = Trainer(nan_resume_args, context=context)
    nan_detected = False
    nan_reason = ""
    try:
        nan_trainer.run()
    except NonFiniteTrainingError as error:
        nan_detected = True
        nan_reason = str(error)
    detected_tensor = torch.tensor(
        int(nan_detected),
        dtype=torch.int32,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(detected_tensor, op=dist.ReduceOp.MIN)
    nan_detected_all_ranks = bool(detected_tensor.item())
    nan_manager = CheckpointManager(str(nan_dir), context)
    nan_latest = nan_manager.find_latest()
    nan_state_unchanged = (
        nan_trainer.state.global_step == 1
        and nan_trainer.scheduler.completed_steps == 1
        and nan_latest is not None
        and nan_latest.name == "step_00000001"
    )

    interrupted_manager = CheckpointManager(
        str(interrupted_dir),
        context,
    )
    interrupted_metadata = interrupted_manager.inspect(
        f"step_{continuous_final_step:08d}"
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
            "detection": "all-rank loss/gradient finite reduction",
            "auto_recovered": False,
            "recovered_checkpoint": nan_latest.name if nan_latest else None,
            "global_step": nan_trainer.state.global_step,
            "tokens_seen": nan_trainer.state.tokens_seen,
            "scheduler_completed_steps": nan_trainer.scheduler.completed_steps,
            "all_ranks_detected": nan_detected_all_ranks,
            "state_unchanged": nan_state_unchanged,
            "status": (
                "detected"
                if nan_detected_all_ranks and nan_state_unchanged
                else "failed"
            ),
            "reason": nan_reason,
        },
        {
            "failure_type": "rank_crash",
            "detection": "launcher exit code / 进程组退出",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "documented_not_injected",
            "reason": "rank crash 需要 launcher 或调度器重启，不在单进程 smoke 中自动修复。",
        },
        {
            "failure_type": "communication_timeout",
            "detection": "NCCL watchdog / doctor connectivity",
            "auto_recovered": False,
            "recovered_checkpoint": None,
            "status": "documented_not_injected",
            "reason": "通信 hang 依赖 NCCL watchdog、超时和多机诊断策略。",
        },
        _half_checkpoint_probe(interrupted_dir)
        if context.is_main
        else {"failure_type": "half_checkpoint", "status": "rank_nonzero"},
    ]

    payload = {
        "benchmark": "fault_tolerance",
        "strategy": args.strategy,
        "precision": args.precision,
        "device": device_name,
        "backend": args.backend,
        "continuous_checkpoint": str(
            continuous_dir / f"step_{continuous_final_step:08d}"
        ),
        "interrupted_checkpoint": str(
            interrupted_dir / f"step_{continuous_final_step:08d}"
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
    if context.is_main:
        _write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    context.barrier()
    if context.initialized_here:
        context.close()
    return payload if context.is_main else None
