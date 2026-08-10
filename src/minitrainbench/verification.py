from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .checkpoint import CheckpointLoad, CheckpointManager
from .distributed import DistributedContext, setup_distributed
from .model import GPTConfig, MiniGPT
from .runtime import TrainingConfig, TrainState, _write_json
from .strategy import create_strategy


def _state_for_comparison(train_state: TrainState) -> dict[str, Any]:
    return {
        "global_step": train_state.global_step,
        "micro_step": train_state.micro_step,
        "tokens_seen": train_state.tokens_seen,
        "seed": train_state.seed,
        "config_fingerprint": train_state.config_fingerprint,
    }


def _update_tensor_digest(hasher: hashlib._Hash, tensor: torch.Tensor) -> None:
    detached = tensor.detach().contiguous()
    hasher.update(str(tuple(detached.shape)).encode())
    hasher.update(str(detached.dtype).encode())
    byte_tensor = detached.reshape(-1).view(torch.uint8).cpu()
    hasher.update(byte_tensor.numpy().tobytes())


def _update_digest(hasher: hashlib._Hash, value: Any) -> None:
    if value.__class__.__name__ == "DTensor" and hasattr(value, "to_local"):
        hasher.update(b"DTensor")
        _update_tensor_digest(hasher, value.to_local())
        return
    if hasattr(value, "local_shards") and callable(value.local_shards):
        hasher.update(f"{value.__class__.__module__}.{value.__class__.__qualname__}".encode())
        shards = value.local_shards()
        for shard in sorted(shards, key=lambda item: repr(item.metadata)):
            _update_digest(hasher, shard.tensor)
        return
    if isinstance(value, torch.Tensor):
        hasher.update(b"Tensor")
        _update_tensor_digest(hasher, value)
        return
    if isinstance(value, Mapping):
        hasher.update(b"Mapping")
        for key in sorted(value, key=repr):
            _update_digest(hasher, key)
            _update_digest(hasher, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        hasher.update(b"Sequence")
        for item in value:
            _update_digest(hasher, item)
        return
    hasher.update(f"{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode())


def _local_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def _distributed_digest(value: Any, context: DistributedContext) -> str:
    local_digest = _local_digest(value)
    if context.world_size == 1:
        return local_digest
    gathered: list[str | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_digest)
    hasher = hashlib.sha256()
    for rank, digest in enumerate(gathered):
        hasher.update(f"{rank}:{digest}".encode())
    return hasher.hexdigest()


def _validate_pair(
    manager: CheckpointManager,
    left: str,
    right: str,
    context: DistributedContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_metadata = manager.inspect(left)
    right_metadata = manager.inspect(right)
    for metadata in (left_metadata, right_metadata):
        if not metadata["ready"]:
            raise ValueError(f"checkpoint 缺少 READY 标记：{metadata['path']}")
        if metadata.get("world_size") != context.world_size:
            raise ValueError(
                f"checkpoint world size={metadata.get('world_size')}，"
                f"当前 torchrun world size={context.world_size}"
            )
    fields = ("strategy", "precision", "world_size", "config_fingerprint")
    mismatches = [
        f"{field}: left={left_metadata.get(field)!r}, right={right_metadata.get(field)!r}"
        for field in fields
        if left_metadata.get(field) != right_metadata.get(field)
    ]
    if mismatches:
        raise ValueError("checkpoint 不能比较，元数据不兼容：" + "；".join(mismatches))
    return left_metadata, right_metadata


def _build_runtime(
    config: TrainingConfig,
    context: DistributedContext,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(config.seed + context.rank)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + context.rank)
    strategy = create_strategy(config.strategy)
    model = MiniGPT(
        GPTConfig(**config.model_dict()),
        activation_checkpointing=config.activation_checkpointing,
    ).to(context.device)
    model = strategy.wrap_model(model, context, config.precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer


def _load_and_digest(
    manager: CheckpointManager,
    path: str,
    config: TrainingConfig,
    context: DistributedContext,
) -> tuple[CheckpointLoad, dict[str, str]]:
    model, optimizer = _build_runtime(config, context)
    current_state = TrainState(
        seed=config.seed,
        config_fingerprint=config.fingerprint(),
    )
    loaded = manager.load(path, model, optimizer, config, current_state)
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    return loaded, {
        "model": _distributed_digest(model_state, context),
        "optimizer": _distributed_digest(optimizer_state, context),
        "train_state": _distributed_digest(
            _state_for_comparison(loaded.train_state),
            context,
        ),
        "rng": _distributed_digest(loaded.rng_state or {}, context),
    }


def verify_checkpoints(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(
        args.backend,
        args.device,
        requires_process_group=True,
    )
    try:
        manager = CheckpointManager(root=None, context=context)
        left_metadata, _right_metadata = _validate_pair(
            manager,
            args.left,
            args.right,
            context,
        )
        config = TrainingConfig.from_dict(left_metadata["config"])
        left_loaded, left_digests = _load_and_digest(
            manager,
            args.left,
            config,
            context,
        )
        right_loaded, right_digests = _load_and_digest(
            manager,
            args.right,
            config,
            context,
        )
        matches = {
            name: left_digests[name] == right_digests[name]
            for name in ("model", "optimizer", "train_state")
        }
        matches["rng"] = (
            left_loaded.deterministic
            and right_loaded.deterministic
            and left_digests["rng"] == right_digests["rng"]
        )
        payload = {
            "benchmark": "checkpoint_verification",
            "left": str(Path(args.left)),
            "right": str(Path(args.right)),
            "strategy": config.strategy,
            "precision": config.precision,
            "world_size": context.world_size,
            "config_fingerprint": config.fingerprint(),
            "left_deterministic": left_loaded.deterministic,
            "right_deterministic": right_loaded.deterministic,
            "left_determinism_reason": left_loaded.determinism_reason,
            "right_determinism_reason": right_loaded.determinism_reason,
            "matches": matches,
            "digests": {
                "left": left_digests,
                "right": right_digests,
            },
            "exact_match": all(matches.values()),
        }
        if context.is_main:
            _write_json(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        return payload if context.is_main else None
    finally:
        context.close()
