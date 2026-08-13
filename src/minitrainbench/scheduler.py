from __future__ import annotations

import math
from typing import Any

import torch


class LearningRateScheduler:
    """以已完成 optimizer step 数为时间轴的轻量学习率调度器。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule: str,
        base_lr: float,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.schedule = schedule
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.min_lr = float(min_lr)
        self.completed_steps = 0
        self.validate()
        self._apply_lr()

    def validate(self) -> None:
        if self.base_lr <= 0:
            raise ValueError("learning-rate 必须大于 0")
        if self.warmup_steps < 0 or self.decay_steps < 0:
            raise ValueError("lr-warmup-steps 和 lr-decay-steps 不能为负数")
        if not 0 <= self.min_lr <= self.base_lr:
            raise ValueError("min-learning-rate 必须位于 [0, learning-rate] 范围内")
        if self.schedule == "constant":
            if self.warmup_steps or self.decay_steps or self.min_lr:
                raise ValueError(
                    "constant scheduler 要求 lr-warmup-steps、lr-decay-steps "
                    "和 min-learning-rate 均为 0"
                )
            return
        if self.schedule != "cosine":
            raise ValueError(f"不支持的 lr scheduler: {self.schedule}")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError(
                "cosine scheduler 要求 lr-decay-steps 大于 lr-warmup-steps"
            )

    def lr_for_step(self, completed_steps: int) -> float:
        if self.schedule == "constant":
            return self.base_lr
        if self.warmup_steps and completed_steps < self.warmup_steps:
            return self.base_lr * (completed_steps + 1) / self.warmup_steps
        if completed_steps >= self.decay_steps:
            return self.min_lr
        progress = (completed_steps - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_lr(self) -> None:
        learning_rate = self.lr_for_step(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> None:
        self.completed_steps += 1
        self._apply_lr()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "completed_steps": torch.tensor(self.completed_steps, dtype=torch.int64)
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        value = state_dict["completed_steps"]
        self.completed_steps = int(value.item() if isinstance(value, torch.Tensor) else value)
        if self.completed_steps < 0:
            raise ValueError("scheduler completed_steps 不能为负数")
        self._apply_lr()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> LearningRateScheduler:
    return LearningRateScheduler(
        optimizer=optimizer,
        schedule=config.lr_scheduler,
        base_lr=config.learning_rate,
        warmup_steps=config.lr_warmup_steps,
        decay_steps=config.lr_decay_steps,
        min_lr=config.min_learning_rate,
    )
