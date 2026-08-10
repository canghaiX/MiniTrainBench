from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .checkpoint import CheckpointManager
from .data import SyntheticTokenIterator
from .distributed import DistributedContext, setup_distributed
from .model import GPTConfig, MiniGPT, count_parameters
from .strategy import TrainingStrategy, create_strategy


def _write_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 benchmark 模式需要 CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备未报告 BF16 支持")
        return torch.bfloat16
    raise ValueError(f"不支持的精度: {precision}")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_max(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    tensor = torch.tensor(value, device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.MAX)
    return float(tensor.item()) if context.is_main else value


def _reduce_mean(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    tensor = torch.tensor(value, device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size) if context.is_main else value


def _summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = [
        "tokens_per_sec",
        "step_time_ms",
        "data_time_ms",
        "forward_backward_ms",
        "optimizer_step_ms",
        "max_cuda_memory_mb",
    ]
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(repeat[metric]) for repeat in repeats]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


@dataclass(frozen=True)
class TrainingConfig:
    strategy: str
    precision: str
    backend: str
    device: str
    activation_checkpointing: bool
    grad_accum_steps: int
    batch_size: int
    seq_length: int
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    learning_rate: float
    steps: int
    warmup_steps: int
    repeat: int
    seed: int
    keep_last: int

    @classmethod
    def from_args(
        cls,
        args: Any,
        context: DistributedContext,
    ) -> TrainingConfig:
        return cls(
            strategy=args.strategy,
            precision=args.precision,
            backend=args.backend or (
                "nccl" if context.device.type == "cuda" else "gloo"
            ),
            device=str(context.device),
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
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            repeat=args.repeat,
            seed=args.seed,
            keep_last=args.keep_last,
        )

    def model_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "precision": self.precision,
            "activation_checkpointing": self.activation_checkpointing,
            "grad_accum_steps": self.grad_accum_steps,
            "batch_size": self.batch_size,
            "model_config": self.model_dict(),
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainState:
    global_step: int = 0
    micro_step: int = 0
    tokens_seen: int = 0
    seed: int = 0
    config_fingerprint: str = ""
    resumed_from: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TrainState:
        return cls(
            global_step=int(state["global_step"]),
            micro_step=int(state["micro_step"]),
            tokens_seen=int(state["tokens_seen"]),
            seed=int(state["seed"]),
            config_fingerprint=str(state["config_fingerprint"]),
            resumed_from=state.get("resumed_from"),
        )


@dataclass
class StepMetrics:
    data_time_ms: float
    forward_backward_ms: float
    optimizer_step_ms: float
    step_time_ms: float
    tokens_per_sec: float
    loss: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class Trainer:
    """最小训练 Runtime：初始化、训练、指标和 checkpoint 生命周期。"""

    def __init__(
        self,
        args: Any,
        context: DistributedContext | None = None,
    ) -> None:
        self.args = args
        self.strategy: TrainingStrategy = create_strategy(args.strategy)
        self.context = context or setup_distributed(
            args.backend,
            args.device,
            requires_process_group=self.strategy.requires_process_group()
            or bool(args.checkpoint_dir)
            or bool(args.resume),
        )
        self._owns_context = context is None
        self.dtype = _precision_dtype(args.precision, self.context.device)
        self.config = TrainingConfig.from_args(args, self.context)
        self.model_config = GPTConfig(**self.config.model_dict())
        self.state = TrainState(
            seed=self.config.seed,
            config_fingerprint=self.config.fingerprint(),
        )
        self.iterator = SyntheticTokenIterator(
            vocab_size=self.config.vocab_size,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            seed=self.config.seed,
            rank=self.context.rank,
        )

        torch.manual_seed(self.config.seed + self.context.rank)
        if self.context.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed + self.context.rank)
        model = MiniGPT(
            self.model_config,
            activation_checkpointing=self.config.activation_checkpointing,
        ).to(self.context.device)
        self.parameter_count = count_parameters(model)
        model = self.strategy.wrap_model(
            model,
            self.context,
            self.config.precision,
        )
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.checkpoint_manager = CheckpointManager(
            root=args.checkpoint_dir,
            context=self.context,
        )
        self.resumed = False
        self.resume_path: str | None = None
        self.last_checkpoint: str | None = None
        self._maybe_resume(args.resume)

    def _maybe_resume(self, resume: str | None) -> None:
        if not resume:
            return
        if not self.checkpoint_manager.enabled:
            raise ValueError("使用 --resume 时必须同时提供 --checkpoint-dir")
        resume_target = resume
        if resume == "latest":
            latest = self.checkpoint_manager.find_latest()
            if latest is None:
                raise ValueError("未找到带 READY 标记的最新 checkpoint")
            resume_target = str(latest)
        loaded_state, metadata = self.checkpoint_manager.load(
            resume_target,
            self.model,
            self.optimizer,
            self.config,
            self.state,
        )
        self.state = loaded_state
        self.iterator.load_state_dict(
            {"seed": self.state.seed, "rank": self.context.rank}
        )
        self.state.resumed_from = str(resume_target)
        self.resumed = True
        self.resume_path = str(resume_target)
        self.last_checkpoint = metadata["path"]

    def _run_one_step(self) -> StepMetrics:
        _sync_device(self.context.device)
        started = time.perf_counter()
        data_started = started
        inputs: list[torch.Tensor] = []
        for micro_index in range(self.config.grad_accum_steps):
            self.state.micro_step = micro_index
            inputs.append(
                self.iterator.batch_for_step(
                    self.state.global_step * self.config.grad_accum_steps + micro_index,
                    self.context.device,
                )
            )
        data_time = time.perf_counter() - data_started

        self.optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        last_loss = 0.0
        for input_ids in inputs:
            with torch.autocast(
                device_type=self.context.device.type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ):
                _, loss = self.model(input_ids, input_ids)
            assert loss is not None
            (loss / self.config.grad_accum_steps).backward()
            last_loss = float(loss.detach().item())
        _sync_device(self.context.device)
        forward_backward = time.perf_counter() - forward_started

        optimizer_started = time.perf_counter()
        self.optimizer.step()
        _sync_device(self.context.device)
        optimizer_step = time.perf_counter() - optimizer_started
        step_time = time.perf_counter() - started

        self.state.global_step += 1
        self.state.micro_step = 0
        self.state.tokens_seen += (
            self.config.batch_size
            * self.config.seq_length
            * self.config.grad_accum_steps
            * self.context.world_size
        )
        return StepMetrics(
            data_time_ms=data_time * 1000,
            forward_backward_ms=forward_backward * 1000,
            optimizer_step_ms=optimizer_step * 1000,
            step_time_ms=step_time * 1000,
            tokens_per_sec=(
                self.config.batch_size
                * self.config.seq_length
                * self.config.grad_accum_steps
                * self.context.world_size
                / step_time
            ),
            loss=last_loss,
        )

    def _run_window(self, warmup_steps: int, measured_steps: int) -> dict[str, Any]:
        if self.context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.context.device)
        self.context.barrier()
        measured: list[StepMetrics] = []
        for step_index in range(warmup_steps + measured_steps):
            metrics = self._run_one_step()
            if self.checkpoint_manager.enabled and self.args.save_every:
                if self.state.global_step % self.args.save_every == 0:
                    self.last_checkpoint = self.checkpoint_manager.save(
                        self.model,
                        self.optimizer,
                        self.state,
                        self.config,
                        keep_last=self.config.keep_last,
                    )
            if step_index >= warmup_steps:
                measured.append(metrics)
        local = {
            "data_time_ms": statistics.fmean(item.data_time_ms for item in measured),
            "forward_backward_ms": statistics.fmean(
                item.forward_backward_ms for item in measured
            ),
            "optimizer_step_ms": statistics.fmean(
                item.optimizer_step_ms for item in measured
            ),
            "step_time_ms": statistics.fmean(item.step_time_ms for item in measured),
            "loss": measured[-1].loss,
        }
        result = {
            key: _reduce_max(value, self.context)
            if key != "loss"
            else _reduce_mean(value, self.context)
            for key, value in local.items()
        }
        result["max_cuda_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(self.context.device) / 1024**2)
            if self.context.device.type == "cuda"
            else 0.0
        )
        result["max_cuda_memory_mb"] = _reduce_max(
            result["max_cuda_memory_mb"],
            self.context,
        )
        tokens_per_step = (
            self.config.batch_size
            * self.config.seq_length
            * self.config.grad_accum_steps
            * self.context.world_size
        )
        result["tokens_per_sec"] = tokens_per_step / (result["step_time_ms"] / 1000)
        return result

    def _result(
        self,
        repeats: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        selected = {
            "tokens_per_sec": summary["tokens_per_sec"]["mean"],
            "step_time_ms": summary["step_time_ms"]["mean"],
            "data_time_ms": summary["data_time_ms"]["mean"],
            "forward_backward_ms": summary["forward_backward_ms"]["mean"],
            "optimizer_step_ms": summary["optimizer_step_ms"]["mean"],
            "max_cuda_memory_mb": summary["max_cuda_memory_mb"]["mean"],
            "loss": repeats[-1]["loss"],
        }
        result: dict[str, Any] = {
            "benchmark": "training",
            "strategy": self.config.strategy,
            "world_size": self.context.world_size,
            "precision": self.config.precision,
            "device": str(self.context.device),
            "backend": self.config.backend,
            "gradient_accumulation_steps": self.config.grad_accum_steps,
            "activation_checkpointing": self.config.activation_checkpointing,
            **selected,
            "repeat_count": len(repeats),
            "parameters": self.parameter_count,
            "steps": self.config.steps,
            "warmup_steps": self.config.warmup_steps,
            "batch_size_per_rank": self.config.batch_size,
            "global_batch_size": (
                self.config.batch_size
                * self.context.world_size
                * self.config.grad_accum_steps
            ),
            "tokens_seen": self.state.tokens_seen,
            "global_step": self.state.global_step,
            "model_config": self.config.model_dict(),
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "runtime": {
                "strategy_impl": self.strategy.name(),
                "checkpoint_dir": self.args.checkpoint_dir,
                "resume": self.resumed,
                "resume_path": self.resume_path,
                "last_checkpoint": self.last_checkpoint,
                "latest_checkpoint": self._latest_checkpoint_name(),
                "keep_last": self.config.keep_last,
                "ready_checkpoints": self._ready_checkpoint_count(),
                "global_step": self.state.global_step,
                "micro_step": self.state.micro_step,
                "tokens_seen": self.state.tokens_seen,
                "seed": self.state.seed,
                "config_fingerprint": self.state.config_fingerprint,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(self.context.local_rank)
                if self.context.device.type == "cuda"
                else None,
            },
        }
        if len(repeats) > 1:
            result["repeats"] = repeats
            result["summary"] = summary
        return result

    def _ready_checkpoint_count(self) -> int:
        if not self.checkpoint_manager.enabled:
            return 0
        return len(self.checkpoint_manager.list_ready())

    def _latest_checkpoint_name(self) -> str | None:
        latest = self.checkpoint_manager.find_latest()
        return latest.name if latest is not None else None

    def run(self) -> dict[str, Any] | None:
        try:
            if self.context.device.type == "cuda":
                torch.cuda.empty_cache()
            self.model.train()
            self.context.barrier()
            repeats: list[dict[str, Any]] = []
            repeat_count = 1 if self.resumed else self.config.repeat
            for repeat_index in range(repeat_count):
                warmup = 0 if self.resumed else self.config.warmup_steps
                metrics = self._run_window(warmup, self.config.steps)
                if self.context.is_main:
                    repeats.append({"repeat_index": repeat_index, **metrics})
            summary = _summarize_repeats(repeats) if self.context.is_main else {}
            result = self._result(repeats, summary) if self.context.is_main else None
            if self.context.is_main and result is not None:
                _write_json(self.args.output, result)
                print(json.dumps(result, indent=2, sort_keys=True))
            return result
        finally:
            if self._owns_context:
                self.context.close()
