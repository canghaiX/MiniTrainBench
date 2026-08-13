from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
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
    def default_gradient_sync_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def wrap_model(
        self,
        model: nn.Module,
        context: DistributedContext,
        precision: str,
    ) -> nn.Module:
        raise NotImplementedError

    def resolve_gradient_sync_mode(self, requested_mode: str) -> str:
        if requested_mode == "auto":
            return self.default_gradient_sync_mode()
        if requested_mode in {"every", "last"}:
            return requested_mode
        raise ValueError(f"不支持的 gradient sync 模式: {requested_mode}")

    def gradient_sync_context(
        self,
        model: nn.Module,
        sync_gradients: bool,
    ) -> AbstractContextManager[None]:
        if sync_gradients or not hasattr(model, "no_sync"):
            return nullcontext()
        return model.no_sync()  # type: ignore[no-any-return]

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
    ) -> torch.Tensor:
        """返回裁剪前的全局梯度范数；max_norm=inf 时只计算不裁剪。"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class DDPStrategy(TrainingStrategy):
    def name(self) -> str:
        return "DDPStrategy"

    def requires_process_group(self) -> bool:
        return False

    def default_gradient_sync_mode(self) -> str:
        return "last"

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

    def default_gradient_sync_mode(self) -> str:
        return "every"

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

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
    ) -> torch.Tensor:
        if not hasattr(model, "clip_grad_norm_"):
            raise TypeError("FSDP strategy 需要 FullyShardedDataParallel 模型")
        return model.clip_grad_norm_(max_norm)  # type: ignore[no-any-return]


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
