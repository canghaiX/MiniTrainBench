from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import DistributedContext
from .model import TransformerBlock


class TrainingStrategy(ABC):
    """训练策略插件接口，用于隔离 DDP/FSDP 包装逻辑。"""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def requires_process_group(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def wrap_model(
        self,
        model: nn.Module,
        context: DistributedContext,
        precision: str,
    ) -> nn.Module:
        raise NotImplementedError


class DDPStrategy(TrainingStrategy):
    def name(self) -> str:
        return "DDPStrategy"

    def requires_process_group(self) -> bool:
        return False

    def wrap_model(
        self,
        model: nn.Module,
        context: DistributedContext,
        precision: str,
    ) -> nn.Module:
        del precision
        if context.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )


class FSDPStrategy(TrainingStrategy):
    def name(self) -> str:
        return "FSDPStrategy"

    def requires_process_group(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        context: DistributedContext,
        precision: str,
    ) -> nn.Module:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        mixed_precision = None
        if precision == "bf16":
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        kwargs: dict[str, Any] = {
            "auto_wrap_policy": policy,
            "mixed_precision": mixed_precision,
            "use_orig_params": True,
        }
        if context.device.type == "cuda":
            kwargs["device_id"] = context.device
        return FSDP(model, **kwargs)


_STRATEGIES: dict[str, type[TrainingStrategy]] = {
    "ddp": DDPStrategy,
    "fsdp": FSDPStrategy,
}


def create_strategy(name: str) -> TrainingStrategy:
    try:
        return _STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"不支持的训练策略: {name}") from None


def registered_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))
