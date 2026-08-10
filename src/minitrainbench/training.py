from __future__ import annotations

from .runtime import (
    StepMetrics,
    TrainState,
    Trainer,
    TrainingConfig,
    _write_json,
)


def train(args):
    return Trainer(args).run()


__all__ = [
    "StepMetrics",
    "TrainState",
    "Trainer",
    "TrainingConfig",
    "_write_json",
    "train",
]
