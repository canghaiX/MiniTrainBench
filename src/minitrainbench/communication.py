from __future__ import annotations

import json
import time
from typing import Any

import torch
import torch.distributed as dist

from .distributed import setup_distributed
from .runtime import _write_json

DEFAULT_OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
SUPPORTED_OPERATIONS = set(DEFAULT_OPERATIONS)
SUPPORTED_ALL_TO_ALL_MODES = ("equal", "uneven", "both")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_operations(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_OPERATIONS)
    operations = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(operations) - SUPPORTED_OPERATIONS)
    if unknown:
        raise ValueError(f"不支持的通信操作: {','.join(unknown)}")
    return operations


def _expand_all_to_all_modes(value: str) -> list[str]:
    if value not in SUPPORTED_ALL_TO_ALL_MODES:
        raise ValueError(f"不支持的 all-to-all split 模式: {value}")
    return ["equal", "uneven"] if value == "both" else [value]


def _all_to_all_splits(
    elements_per_peer: int,
    world_size: int,
    rank: int,
    mode: str,
) -> tuple[list[int], list[int]]:
    if mode == "equal":
        input_splits = [elements_per_peer for _ in range(world_size)]
        output_splits = [elements_per_peer for _ in range(world_size)]
    elif mode == "uneven":
        input_splits = [
            elements_per_peer + ((rank + peer) % world_size)
            for peer in range(world_size)
        ]
        output_splits = [
            elements_per_peer + ((peer + rank) % world_size)
            for peer in range(world_size)
        ]
    else:
        raise ValueError(f"不支持的 all-to-all split 模式: {mode}")
    return input_splits, output_splits


def _operation_bytes(
    operation: str,
    elements: int,
    element_size: int,
    world_size: int,
    input_splits: list[int] | None,
) -> int:
    if operation == "all_reduce":
        return elements * element_size
    if operation in {"all_gather", "reduce_scatter"}:
        return elements * world_size * element_size
    if operation == "all_to_all":
        if input_splits is None:
            raise ValueError("all_to_all 缺少 input_splits")
        return sum(input_splits) * element_size
    raise ValueError(f"不支持的通信操作: {operation}")


def _run_operation(
    operation: str,
    tensor: torch.Tensor,
    world_size: int,
    *,
    input_splits: list[int] | None = None,
    output_splits: list[int] | None = None,
) -> None:
    if operation == "all_reduce":
        dist.all_reduce(tensor)
    elif operation == "all_gather":
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
    elif operation == "reduce_scatter":
        output = torch.empty(
            tensor.numel() // world_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(output, tensor)
        else:
            chunks = list(tensor.chunk(world_size))
            dist.reduce_scatter(output, chunks)
    elif operation == "all_to_all":
        if input_splits is None or output_splits is None:
            raise ValueError("all_to_all 需要 input_splits 和 output_splits")
        if not hasattr(dist, "all_to_all_single"):
            raise NotImplementedError("当前 PyTorch 不支持 all_to_all_single")
        output = torch.empty(
            sum(output_splits),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
    else:
        raise ValueError(f"不支持的通信操作: {operation}")


def _run_one_case(
    context: Any,
    operation: str,
    size: int,
    warmup: int,
    iters: int,
    split_mode: str | None,
) -> dict[str, Any] | None:
    input_splits = None
    output_splits = None
    if operation == "reduce_scatter":
        tensor_size = size * context.world_size
    elif operation == "all_to_all":
        if split_mode is None:
            raise ValueError("all_to_all 需要 split_mode")
        input_splits, output_splits = _all_to_all_splits(
            size,
            context.world_size,
            context.rank,
            split_mode,
        )
        tensor_size = sum(input_splits)
    else:
        tensor_size = size

    tensor = torch.ones(
        tensor_size,
        dtype=torch.float32,
        device=context.device,
    )
    try:
        context.barrier()
        for _ in range(warmup):
            _run_operation(
                operation,
                tensor,
                context.world_size,
                input_splits=input_splits,
                output_splits=output_splits,
            )
        _synchronize(context.device)
        started = time.perf_counter()
        for _ in range(iters):
            _run_operation(
                operation,
                tensor,
                context.world_size,
                input_splits=input_splits,
                output_splits=output_splits,
            )
        _synchronize(context.device)
        elapsed = (time.perf_counter() - started) / iters
        gathered_elapsed = torch.tensor(elapsed, device=context.device)
        dist.reduce(gathered_elapsed, dst=0, op=dist.ReduceOp.MAX)
        local_bytes = _operation_bytes(
            operation,
            size,
            tensor.element_size(),
            context.world_size,
            input_splits,
        )
        gathered_bytes = torch.tensor(
            local_bytes,
            dtype=torch.float64,
            device=context.device,
        )
        dist.reduce(gathered_bytes, dst=0, op=dist.ReduceOp.MAX)
        if not context.is_main:
            return None
        operation_bytes = int(gathered_bytes.item())
        result = {
            "operation": operation,
            "elements": size,
            "bytes": operation_bytes,
            "world_size": context.world_size,
            "latency_ms": float(gathered_elapsed.item() * 1000),
            "bandwidth_gbps": float(operation_bytes / gathered_elapsed.item() / 1e9),
            "status": "ok",
        }
    except (RuntimeError, NotImplementedError, ValueError) as error:
        if not context.is_main:
            return None
        result = {
            "operation": operation,
            "elements": size,
            "world_size": context.world_size,
            "status": "skipped",
            "reason": str(error),
        }

    if operation == "all_to_all":
        result.update(
            {
                "split_mode": split_mode,
                "elements_per_peer": size,
                "total_elements_per_rank": tensor_size,
                "input_splits": input_splits,
                "output_splits": output_splits,
            }
        )
    return result


def communication_benchmark(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device)
    try:
        if context.world_size < 2:
            raise ValueError("通信 benchmark 至少需要 2 个进程")
        sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
        operations = _parse_operations(getattr(args, "operations", None))
        all_to_all_modes = _expand_all_to_all_modes(args.all_to_all_mode)
        results: list[dict[str, Any]] = []
        for size in sizes:
            for operation in operations:
                split_modes = all_to_all_modes if operation == "all_to_all" else [None]
                for split_mode in split_modes:
                    result = _run_one_case(
                        context,
                        operation,
                        size,
                        args.warmup,
                        args.iters,
                        split_mode,
                    )
                    if result is not None:
                        results.append(result)
        payload = {
            "benchmark": "communication",
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "world_size": context.world_size,
            "dtype": "float32",
            "sizes": sizes,
            "operations": operations,
            "all_to_all_mode": args.all_to_all_mode,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": results,
        }
        if context.is_main:
            _write_json(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        return payload if context.is_main else None
    finally:
        context.close()
