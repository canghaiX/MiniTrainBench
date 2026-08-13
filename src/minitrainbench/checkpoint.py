from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from .distributed import DistributedContext


@dataclass(frozen=True)
class CheckpointLoad:
    train_state: Any
    metadata: dict[str, Any]
    rng_state: dict[str, torch.Tensor] | None
    deterministic: bool
    determinism_reason: str


class CheckpointManager:
    """封装 DCP 保存、READY 标记、发现和兼容性校验。"""

    def __init__(self, root: str | None, context: DistributedContext) -> None:
        self.root = Path(root) if root else None
        self.context = context

    @property
    def enabled(self) -> bool:
        return self.root is not None

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def _rng_state_path(self, path: Path) -> Path:
        return path / f"rng_state_rank_{self.context.rank:05d}.pt"

    def _resolve_path(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute() and self.root is not None:
            checkpoint_path = self.root / checkpoint_path
        return checkpoint_path

    @staticmethod
    def _metadata_markdown(metadata: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# MiniTrainBench Checkpoint 元数据",
                "",
                f"- Step：{metadata['step']}",
                f"- Strategy：{metadata['strategy']}",
                f"- Precision：{metadata['precision']}",
                f"- World size：{metadata['world_size']}",
                f"- Tokens seen：{metadata['tokens_seen']}",
                f"- 配置指纹：`{metadata['config_fingerprint']}`",
                f"- RNG 状态版本：{metadata.get('rng_state_version', '无（旧 checkpoint）')}",
                f"- Scheduler 状态版本：{metadata.get('scheduler_state_version', '无（旧 checkpoint）')}",
                f"- 生成时间：{metadata['created_at']}",
                "",
                (
                    "只有同 strategy、同 precision、同 world size、同模型配置和同关键训练参数的"
                    "任务可以恢复这个 checkpoint。v3 checkpoint 同时保存 scheduler 与每 "
                    "rank RNG 状态，可验证 optimizer step 时间轴和随机训练路径。"
                ),
            ]
        ) + "\n"

    def _validate_metadata(
        self,
        metadata: dict[str, Any],
        config: Any,
        path: Path,
    ) -> None:
        format_version = int(metadata.get("format_version", 1))
        stored_fingerprint = metadata.get("config_fingerprint")
        fingerprint_matches = stored_fingerprint == config.fingerprint()
        if format_version < 3 and config.uses_legacy_runtime_defaults():
            fingerprint_matches = fingerprint_matches or stored_fingerprint in {
                config.v2_fingerprint(),
                config.legacy_fingerprint(),
            }
        expected = {
            "strategy": config.strategy,
            "precision": config.precision,
            "world_size": self.context.world_size,
            "config_fingerprint": config.fingerprint(),
        }
        mismatches = [
            f"{key}: checkpoint={metadata.get(key)!r}, 当前={value!r}"
            for key, value in expected.items()
            if metadata.get(key) != value
            and not (key == "config_fingerprint" and fingerprint_matches)
        ]
        if mismatches:
            raise ValueError(
                f"checkpoint {path} 与当前训练配置不匹配：" + "；".join(mismatches)
            )

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_state: Any,
        config: Any,
        rng_state: dict[str, torch.Tensor],
        keep_last: int = 3,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("checkpoint 未启用")
        assert self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self.root / f"step_{train_state.global_step:08d}"
        temporary_path = self.root / f".step_{train_state.global_step:08d}.tmp"
        if self.context.is_main:
            shutil.rmtree(temporary_path, ignore_errors=True)
            shutil.rmtree(final_path, ignore_errors=True)
        self.context.barrier()

        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler,
            "train_state": {
                "global_step": torch.tensor(train_state.global_step, dtype=torch.int64),
                "micro_step": torch.tensor(train_state.micro_step, dtype=torch.int64),
                "tokens_seen": torch.tensor(train_state.tokens_seen, dtype=torch.int64),
                "seed": torch.tensor(train_state.seed, dtype=torch.int64),
            },
        }
        dcp.save(state, checkpoint_id=temporary_path)
        torch.save(rng_state, self._rng_state_path(temporary_path))
        self.context.barrier()
        if self.context.is_main:
            metadata = {
                "format_version": 3,
                "rng_state_version": 1,
                "scheduler_state_version": 1,
                "path": str(final_path),
                "step": train_state.global_step,
                "strategy": config.strategy,
                "precision": config.precision,
                "world_size": self.context.world_size,
                "config": config.to_dict(),
                "model_config": config.model_dict(),
                "config_fingerprint": config.fingerprint(),
                "tokens_seen": train_state.tokens_seen,
                "resumed_from": train_state.resumed_from,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._metadata_path(temporary_path).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
            (temporary_path / "metadata_zh.md").write_text(
                self._metadata_markdown(metadata)
            )
            (temporary_path / "READY").write_text("ready\n")
            os.replace(temporary_path, final_path)
            latest_tmp = self.root / ".latest.tmp"
            latest_tmp.write_text(final_path.name + "\n")
            os.replace(latest_tmp, self.root / "latest")
            self.prune(keep_last)
        self.context.barrier()
        return str(final_path)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Any,
        current_state: Any,
    ) -> CheckpointLoad:
        checkpoint_path = self._resolve_path(path)
        if not (checkpoint_path / "READY").is_file():
            raise ValueError(f"checkpoint 缺少 READY 标记：{checkpoint_path}")
        metadata = json.loads(self._metadata_path(checkpoint_path).read_text())
        self._validate_metadata(metadata, config, checkpoint_path)

        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        train_state = {
            "global_step": torch.empty((), dtype=torch.int64, device=self.context.device),
            "micro_step": torch.empty((), dtype=torch.int64, device=self.context.device),
            "tokens_seen": torch.empty((), dtype=torch.int64, device=self.context.device),
            "seed": torch.empty((), dtype=torch.int64, device=self.context.device),
        }
        checkpoint_state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "train_state": train_state,
        }
        format_version = int(metadata.get("format_version", 1))
        if format_version >= 3:
            checkpoint_state["scheduler"] = scheduler
        dcp.load(checkpoint_state, checkpoint_id=checkpoint_path)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
        )
        loaded_state = type(current_state)(
            global_step=int(train_state["global_step"].item()),
            micro_step=int(train_state["micro_step"].item()),
            tokens_seen=int(train_state["tokens_seen"].item()),
            seed=int(train_state["seed"].item()),
            config_fingerprint=str(metadata["config_fingerprint"]),
            resumed_from=str(checkpoint_path),
        )
        if format_version < 3:
            scheduler.load_state_dict(
                {"completed_steps": torch.tensor(loaded_state.global_step)}
            )
        rng_state, deterministic, determinism_reason = self._load_rng_state(checkpoint_path)
        self.context.barrier()
        return CheckpointLoad(
            train_state=loaded_state,
            metadata=metadata,
            rng_state=rng_state,
            deterministic=deterministic,
            determinism_reason=determinism_reason,
        )

    def _load_rng_state(
        self,
        checkpoint_path: Path,
    ) -> tuple[dict[str, torch.Tensor] | None, bool, str]:
        path = self._rng_state_path(checkpoint_path)
        if not path.is_file():
            message = (
                f"checkpoint {checkpoint_path} 缺少 rank {self.context.rank} 的 RNG 状态；"
                "本次恢复可继续训练，但无法保证精确复现。"
            )
            if self.context.is_main:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            return None, False, "checkpoint_missing_rng_state"
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("cpu"), torch.Tensor):
            raise TypeError(f"checkpoint RNG 状态格式无效：{path}")
        rng_state = {
            name: value
            for name, value in loaded.items()
            if isinstance(name, str) and isinstance(value, torch.Tensor)
        }
        if self.context.device.type == "cuda" and "cuda" not in rng_state:
            raise ValueError(f"checkpoint 缺少 CUDA RNG 状态：{path}")
        return rng_state, True, "restored_rank_rng_state"

    def find_latest(self) -> Path | None:
        if not self.enabled:
            return None
        assert self.root is not None
        latest = self.root / "latest"
        if latest.is_file():
            raw = latest.read_text().strip()
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.root / raw
            if (candidate / "READY").is_file():
                return candidate
        candidates = self.list_ready()
        return candidates[-1] if candidates else None

    def list_ready(self) -> list[Path]:
        if not self.enabled:
            return []
        assert self.root is not None
        return sorted(
            path
            for path in self.root.glob("step_*")
            if path.is_dir() and (path / "READY").is_file()
        )

    def inspect(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = self._resolve_path(path)
        metadata_path = self._metadata_path(checkpoint_path)
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
        return {
            **metadata,
            "path": str(checkpoint_path),
            "name": checkpoint_path.name,
            "ready": (checkpoint_path / "READY").is_file(),
            "rng_state_available": self._rng_state_path(checkpoint_path).is_file(),
        }

    def prune(self, keep_last: int) -> None:
        if keep_last <= 0:
            return
        ready = self.list_ready()
        stale = ready[:-keep_last]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
