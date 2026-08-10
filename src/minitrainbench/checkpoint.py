from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from .distributed import DistributedContext


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
                f"- 生成时间：{metadata['created_at']}",
                "",
                "只有同 strategy、同 precision、同 world size、同模型配置和同关键训练参数的"
                "任务可以恢复这个 checkpoint。",
            ]
        ) + "\n"

    def _validate_metadata(
        self,
        metadata: dict[str, Any],
        config: Any,
        path: Path,
    ) -> None:
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
        ]
        if mismatches:
            raise ValueError(
                f"checkpoint {path} 与当前训练配置不匹配：" + "；".join(mismatches)
            )

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_state: Any,
        config: Any,
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
            "train_state": {
                "global_step": torch.tensor(train_state.global_step, dtype=torch.int64),
                "micro_step": torch.tensor(train_state.micro_step, dtype=torch.int64),
                "tokens_seen": torch.tensor(train_state.tokens_seen, dtype=torch.int64),
                "seed": torch.tensor(train_state.seed, dtype=torch.int64),
            },
        }
        dcp.save(state, checkpoint_id=temporary_path)
        self.context.barrier()
        if self.context.is_main:
            metadata = {
                "format_version": 1,
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
        self.context.barrier()
        return str(final_path)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        current_state: Any,
    ) -> tuple[Any, dict[str, Any]]:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_dir() and self.root is not None:
            checkpoint_path = self.root / path
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
        dcp.load(
            {
                "model": model_state,
                "optimizer": optimizer_state,
                "train_state": train_state,
            },
            checkpoint_id=checkpoint_path,
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
        )
        loaded = type(current_state)(
            global_step=int(train_state["global_step"].item()),
            micro_step=int(train_state["micro_step"].item()),
            tokens_seen=int(train_state["tokens_seen"].item()),
            seed=int(train_state["seed"].item()),
            config_fingerprint=str(metadata["config_fingerprint"]),
            resumed_from=str(checkpoint_path),
        )
        self.context.barrier()
        return loaded, metadata

    def find_latest(self) -> Path | None:
        if not self.enabled:
            return None
        assert self.root is not None
        latest = self.root / "latest"
        if latest.is_file():
            candidate = self.root / latest.read_text().strip()
            if (candidate / "READY").is_file():
                return candidate
        candidates = sorted(
            path
            for path in self.root.glob("step_*")
            if path.is_dir() and (path / "READY").is_file()
        )
        return candidates[-1] if candidates else None
