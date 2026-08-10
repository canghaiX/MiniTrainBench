from __future__ import annotations

import json
import time
from typing import Any

import torch
import torch.distributed as dist

from .distributed import setup_distributed
from .training import _write_json


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_operation(
    operation: str,
    tensor: torch.Tensor,
    world_size: int,
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
    else:
        raise ValueError(f"不支持的通信操作: {operation}")


def communication_benchmark(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device)
    try:
        if context.world_size < 2:
            raise ValueError("通信 benchmark 至少需要 2 个进程")
        sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
        results: list[dict[str, Any]] = []
        operations = ["all_reduce", "all_gather", "reduce_scatter"]
        for size in sizes:
            for operation in operations:
                tensor_size = size * context.world_size if operation == "reduce_scatter" else size
                tensor = torch.ones(
                    tensor_size,
                    dtype=torch.float32,
                    device=context.device,
                )
                try:
                    context.barrier()
                    for _ in range(args.warmup):
                        _run_operation(operation, tensor, context.world_size)
                    _synchronize(context.device)
                    started = time.perf_counter()
                    for _ in range(args.iters):
                        _run_operation(operation, tensor, context.world_size)
                    _synchronize(context.device)
                    elapsed = (time.perf_counter() - started) / args.iters
                    gathered_elapsed = torch.tensor(elapsed, device=context.device)
                    dist.reduce(gathered_elapsed, dst=0, op=dist.ReduceOp.MAX)
                    if context.is_main:
                        operation_bytes = size * tensor.element_size()
                        if operation != "all_reduce":
                            operation_bytes *= context.world_size
                        results.append(
                            {
                                "operation": operation,
                                "elements": size,
                                "bytes": operation_bytes,
                                "world_size": context.world_size,
                                "latency_ms": float(gathered_elapsed.item() * 1000),
                                "bandwidth_gbps": float(
                                    operation_bytes / gathered_elapsed.item() / 1e9
                                ),
                                "status": "ok",
                            }
                        )
                except (RuntimeError, NotImplementedError, ValueError) as error:
                    if context.is_main:
                        results.append(
                            {
                                "operation": operation,
                                "elements": size,
                                "world_size": context.world_size,
                                "status": "skipped",
                                "reason": str(error),
                            }
                        )
        payload = {
            "benchmark": "communication",
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "world_size": context.world_size,
            "dtype": "float32",
            "sizes": sizes,
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
