from __future__ import annotations

import json
import math
import time
from typing import Any

import torch
import torch.distributed as dist

from .distributed import setup_distributed
from .runtime import _write_json


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_balance_loss(counts: torch.Tensor) -> float:
    total = float(counts.sum().item())
    if total <= 0:
        return 0.0
    probs = counts.float() / total
    target = torch.full_like(probs, 1.0 / counts.numel())
    return float(((probs - target) ** 2).sum().item() * counts.numel())


def _pack_by_owner(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_owners: torch.Tensor,
    num_ranks: int,
    capacity: int,
) -> tuple[torch.Tensor, list[int], list[int], torch.Tensor]:
    accepted_mask = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    expert_counts = torch.zeros(expert_owners.numel(), dtype=torch.int64, device=tokens.device)
    send_counts = [0 for _ in range(num_ranks)]
    order: list[torch.Tensor] = []
    for expert_id in range(expert_owners.numel()):
        token_indices = torch.nonzero(expert_ids == expert_id, as_tuple=False).flatten()
        if token_indices.numel() == 0:
            continue
        keep = token_indices[:capacity]
        accepted_mask[keep] = True
        expert_counts[expert_id] = int(keep.numel())
        owner = int(expert_owners[expert_id].item())
        send_counts[owner] += int(keep.numel())
        order.append(keep)
    if order:
        flat_indices = torch.cat(order, dim=0)
        packed = tokens[flat_indices].contiguous()
    else:
        flat_indices = torch.empty(0, dtype=torch.int64, device=tokens.device)
        packed = tokens.new_empty((0, tokens.shape[-1]))
    return packed, send_counts, expert_counts.tolist(), accepted_mask


def _all_to_all_dispatch(
    context: Any,
    local_tokens: torch.Tensor,
    send_counts: list[int],
) -> tuple[torch.Tensor, list[int], float]:
    hidden_size = local_tokens.shape[-1]
    gathered_counts: list[list[int]] | None = None
    if context.world_size > 1:
        gathered_counts = [[] for _ in range(context.world_size)]
        dist.all_gather_object(gathered_counts, send_counts)
    if context.world_size == 1:
        return local_tokens, send_counts, 0.0

    assert gathered_counts is not None
    recv_counts_by_source = [int(gathered_counts[source][context.rank]) for source in range(context.world_size)]
    input_splits = [count * hidden_size for count in send_counts]
    output_splits = [count * hidden_size for count in recv_counts_by_source]
    send_flat = local_tokens.reshape(-1).contiguous()
    recv_flat = torch.empty(sum(output_splits), dtype=local_tokens.dtype, device=local_tokens.device)
    _synchronize(context.device)
    started = time.perf_counter()
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    _synchronize(context.device)
    elapsed = (time.perf_counter() - started) * 1000
    return recv_flat.reshape(-1, hidden_size), recv_counts_by_source, elapsed


def _all_to_all_combine(
    context: Any,
    local_tokens: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
) -> float:
    hidden_size = local_tokens.shape[-1]
    if context.world_size == 1:
        return 0.0
    input_splits = [count * hidden_size for count in recv_counts]
    output_splits = [count * hidden_size for count in send_counts]
    send_flat = local_tokens.reshape(-1).contiguous()
    recv_flat = torch.empty(sum(output_splits), dtype=local_tokens.dtype, device=local_tokens.device)
    _synchronize(context.device)
    started = time.perf_counter()
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    _synchronize(context.device)
    return (time.perf_counter() - started) * 1000


def moe_routing_benchmark(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device)
    try:
        if args.num_experts < 1:
            raise ValueError("num-experts 必须为正数")
        if args.hidden_size < 1 or args.tokens_per_rank < 1:
            raise ValueError("hidden-size 和 tokens-per-rank 必须为正数")
        if args.capacity_factor <= 0:
            raise ValueError("capacity-factor 必须大于 0")
        if args.num_experts % context.world_size != 0 and context.world_size > 1:
            raise ValueError("num-experts 必须能被 world size 整除，便于 expert parallel 演示")

        generator = torch.Generator(device="cpu").manual_seed(args.seed + context.rank)
        tokens = torch.randn(
            args.tokens_per_rank,
            args.hidden_size,
            generator=generator,
        ).to(context.device)
        router = torch.randn(
            args.num_experts,
            args.hidden_size,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        ).to(context.device)
        logits = tokens @ router.t()
        expert_ids = torch.argmax(logits, dim=-1)
        owner_map = torch.arange(args.num_experts, device=context.device) % context.world_size
        capacity = max(
            1,
            math.ceil(
                args.tokens_per_rank
                * context.world_size
                / args.num_experts
                * args.capacity_factor
            ),
        )
        packed, send_counts, local_expert_counts, accepted_mask = _pack_by_owner(
            tokens,
            expert_ids,
            owner_map,
            context.world_size,
            capacity,
        )
        local_dropped = int((~accepted_mask).sum().item())

        total_expert_counts = torch.tensor(
            local_expert_counts,
            dtype=torch.int64,
            device=context.device,
        )
        if context.world_size > 1:
            dist.all_reduce(total_expert_counts, op=dist.ReduceOp.SUM)
            dropped_tensor = torch.tensor(local_dropped, dtype=torch.int64, device=context.device)
            dist.all_reduce(dropped_tensor, op=dist.ReduceOp.SUM)
            total_dropped = int(dropped_tensor.item())
        else:
            total_dropped = local_dropped

        dispatch_time_ms = 0.0
        combine_time_ms = 0.0
        dispatched = packed
        recv_counts: list[int] = [0 for _ in range(context.world_size)]
        if context.world_size > 1:
            try:
                dispatched, recv_counts, dispatch_time_ms = _all_to_all_dispatch(
                    context,
                    packed,
                    send_counts,
                )
                processed = torch.tanh(dispatched)
                combine_time_ms = _all_to_all_combine(
                    context,
                    processed,
                    send_counts,
                    recv_counts,
                )
            except (RuntimeError, NotImplementedError) as error:
                if not context.is_main:
                    return None
                payload = {
                    "benchmark": "moe_routing",
                    "status": "skipped",
                    "reason": str(error),
                    "world_size": context.world_size,
                    "num_experts": args.num_experts,
                }
                _write_json(args.output, payload)
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
                return payload
        else:
            processed = torch.tanh(dispatched)

        load_imbalance = float(
            total_expert_counts.float().max().item() / max(1.0, total_expert_counts.float().mean().item())
        )
        bytes_per_direction = int(
            sum(send_counts) * args.hidden_size * processed.element_size()
        )
        payload = {
            "benchmark": "moe_routing",
            "status": "ok",
            "world_size": context.world_size,
            "device": str(context.device),
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "num_experts": args.num_experts,
            "top_k": 1,
            "capacity_factor": args.capacity_factor,
            "tokens_per_rank": args.tokens_per_rank,
            "hidden_size": args.hidden_size,
            "capacity_per_expert": capacity,
            "total_tokens": int(context.world_size * args.tokens_per_rank),
            "expert_token_counts": [int(value) for value in total_expert_counts.cpu().tolist()],
            "tokens_dropped": total_dropped,
            "load_imbalance_ratio": load_imbalance,
            "load_balance_loss": _load_balance_loss(total_expert_counts),
            "dispatch_time_ms": dispatch_time_ms,
            "combine_time_ms": combine_time_ms,
            "communication_bytes": bytes_per_direction * (2 if context.world_size > 1 else 0),
            "send_counts_by_rank": send_counts,
            "recv_counts_by_rank": recv_counts,
            "expert_owner_map": [int(value) for value in owner_map.cpu().tolist()],
        }
        if context.is_main:
            _write_json(args.output, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload if context.is_main else None
    finally:
        context.close()
