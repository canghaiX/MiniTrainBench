from __future__ import annotations

from .runtime import (
    StepMetrics,
    Trainer,
    TrainingConfig,
    TrainState,
    _write_json,
)
from .strategy import DDPStrategy, FSDPStrategy, TrainingStrategy, create_strategy


def train(args):
    return Trainer(args).run()


__all__ = [
    "DDPStrategy",
    "FSDPStrategy",
    "StepMetrics",
    "TrainState",
    "Trainer",
    "TrainingConfig",
    "TrainingStrategy",
    "_write_json",
    "create_strategy",
    "train",
]
