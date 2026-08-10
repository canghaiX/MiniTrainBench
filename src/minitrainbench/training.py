from __future__ import annotations

from .runtime import (
    StepMetrics,
    TrainState,
    Trainer,
    TrainingConfig,
    _write_json,
)
from .strategy import DDPStrategy, FSDPStrategy, TrainingStrategy, create_strategy


def train(args):
    return Trainer(args).run()


__all__ = [
    "StepMetrics",
    "TrainState",
    "Trainer",
    "TrainingConfig",
    "TrainingStrategy",
    "DDPStrategy",
    "FSDPStrategy",
    "create_strategy",
    "_write_json",
    "train",
]
