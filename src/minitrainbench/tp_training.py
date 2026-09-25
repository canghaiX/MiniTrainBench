from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .distributed import DistributedContext, setup_distributed
from .runtime import _write_json


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        values: torch.Tensor,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        context.process_group = process_group
        output = values.clone()
        dist.all_reduce(output, group=process_group)
        return output

    @staticmethod
    def backward(
        context: Any,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        output = gradient.clone()
        dist.all_reduce(output, group=context.process_group)
        return output, None


def _all_reduce_sum(values: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    return _AllReduceSum.apply(values, process_group)


@dataclass(frozen=True)
class ParallelGroups:
    tensor: dist.ProcessGroup
    data: dist.ProcessGroup
    tensor_rank: int
    data_rank: int
    tensor_size: int
    data_size: int


def _build_parallel_groups(
    context: DistributedContext,
    tensor_size: int,
) -> ParallelGroups:
    if tensor_size < 1 or context.world_size % tensor_size != 0:
        raise ValueError(
            f"tp-size 必须是 world size 的正整数因子，当前 tp-size={tensor_size}、"
            f"world-size={context.world_size}"
        )
    data_size = context.world_size // tensor_size
    tensor_group = None
    data_group = None
    for data_rank in range(data_size):
        ranks = list(range(data_rank * tensor_size, (data_rank + 1) * tensor_size))
        group = dist.new_group(ranks=ranks)
        if context.rank in ranks:
            tensor_group = group
    for tensor_rank in range(tensor_size):
        ranks = [data_rank * tensor_size + tensor_rank for data_rank in range(data_size)]
        group = dist.new_group(ranks=ranks)
        if context.rank in ranks:
            data_group = group
    assert tensor_group is not None
    assert data_group is not None
    return ParallelGroups(
        tensor=tensor_group,
        data=data_group,
        tensor_rank=context.rank % tensor_size,
        data_rank=context.rank // tensor_size,
        tensor_size=tensor_size,
        data_size=data_size,
    )


def _reference_parameters(
    in_features: int,
    hidden_features: int,
    out_features: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(hidden_features, in_features, generator=generator),
        torch.randn(hidden_features, generator=generator),
        torch.randn(out_features, hidden_features, generator=generator),
        torch.randn(out_features, generator=generator),
    )


def _batch_for_step(
    batch_size: int,
    sequence_length: int,
    in_features: int,
    out_features: int,
    seed: int,
    data_rank: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + data_rank * 100_003 + step * 1_000_003
    )
    shape = (batch_size * sequence_length, in_features)
    inputs = torch.randn(*shape, generator=generator).to(device)
    targets = torch.randn(
        batch_size * sequence_length,
        out_features,
        generator=generator,
    ).to(device)
    return inputs, targets


class TensorParallelMLP(nn.Module):
    def __init__(
        self,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        weight_one, bias_one, weight_two, bias_two = weights
        hidden_features, in_features = weight_one.shape
        out_features = weight_two.shape[0]
        if hidden_features % tensor_size != 0:
            raise ValueError("hidden-features 必须能被 tp-size 整除")
        local_hidden = hidden_features // tensor_size
        start = tensor_rank * local_hidden
        end = start + local_hidden
        self.weight_one = nn.Parameter(weight_one[start:end].to(device).clone())
        self.bias_one = nn.Parameter(bias_one[start:end].to(device).clone())
        self.weight_two = nn.Parameter(weight_two[:, start:end].to(device).clone())
        self.bias_two = nn.Parameter(bias_two.to(device).clone())
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.local_hidden = local_hidden

    def forward(
        self,
        inputs: torch.Tensor,
        tensor_group: dist.ProcessGroup,
        autocast_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            hidden = F.gelu(F.linear(inputs, self.weight_one, self.bias_one))
            partial_output = F.linear(hidden, self.weight_two, None)
            output = _all_reduce_sum(partial_output, tensor_group)
            return output + self.bias_two


def _reduce_max(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.MAX)
    return float(tensor.item()) if context.is_main else value


def _sync_data_gradients(model: nn.Module, groups: ParallelGroups) -> None:
    if groups.data_size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, group=groups.data)
        parameter.grad.div_(groups.data_size)


def _sync_tensor_replicated_gradients(
    model: TensorParallelMLP,
    groups: ParallelGroups,
) -> None:
    if model.bias_two.grad is None or groups.tensor_size == 1:
        return
    dist.all_reduce(model.bias_two.grad, group=groups.tensor)


def _sync_reference_gradients(
    reference_weights: tuple[torch.Tensor, ...],
    groups: ParallelGroups,
) -> None:
    if groups.data_size == 1:
        return
    for parameter in reference_weights:
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, group=groups.data)
        parameter.grad.div_(groups.data_size)


def _reference_forward(
    inputs: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    weight_one, bias_one, weight_two, bias_two = weights
    with torch.autocast(
        device_type=inputs.device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        hidden = F.gelu(F.linear(inputs, weight_one, bias_one))
        return F.linear(hidden, weight_two, bias_two)


def _correctness_probe(
    model: TensorParallelMLP,
    groups: ParallelGroups,
    context: DistributedContext,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    args: Any,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float | str]:
    inputs, targets = _batch_for_step(
        args.batch_size,
        args.seq_length,
        args.in_features,
        args.out_features,
        args.seed,
        groups.data_rank,
        0,
        context.device,
    )
    reference_weights = tuple(
        weight.to(context.device).detach().requires_grad_(True) for weight in weights
    )
    reference_output = _reference_forward(inputs, reference_weights, autocast_dtype)
    model.zero_grad(set_to_none=True)
    output = model(inputs, groups.tensor, autocast_dtype)
    loss = F.mse_loss(output.float(), targets.float())
    reference_loss = F.mse_loss(reference_output.float(), targets.float())
    (loss / groups.tensor_size).backward()
    _sync_tensor_replicated_gradients(model, groups)
    _sync_data_gradients(model, groups)
    reference_loss.backward()
    _sync_reference_gradients(reference_weights, groups)
    weight_one, bias_one, weight_two, bias_two = reference_weights
    local_hidden_start = groups.tensor_rank * model.local_hidden
    local_hidden_end = local_hidden_start + model.local_hidden
    errors = {
        "forward_max_error": float(
            (output.float() - reference_output.float()).abs().max().item()
        ),
        "loss_error": float(abs(loss.item() - reference_loss.item())),
        "weight_one_grad_error": float(
            (
                model.weight_one.grad.float()
                - weight_one.grad[local_hidden_start:local_hidden_end].float()
            )
            .abs()
            .max()
            .item()
        ),
        "bias_one_grad_error": float(
            (
                model.bias_one.grad.float()
                - bias_one.grad[local_hidden_start:local_hidden_end].float()
            )
            .abs()
            .max()
            .item()
        ),
        "weight_two_grad_error": float(
            (
                model.weight_two.grad.float()
                - weight_two.grad[:, local_hidden_start:local_hidden_end].float()
            )
            .abs()
            .max()
            .item()
        ),
        "bias_two_grad_error": float(
            (model.bias_two.grad.float() - bias_two.grad.float()).abs().max().item()
        ),
    }
    comparisons = (
        (output.float(), reference_output.float()),
        (loss.detach().reshape(1), reference_loss.detach().reshape(1)),
        (
            model.weight_one.grad.float(),
            weight_one.grad[local_hidden_start:local_hidden_end].float(),
        ),
        (
            model.bias_one.grad.float(),
            bias_one.grad[local_hidden_start:local_hidden_end].float(),
        ),
        (
            model.weight_two.grad.float(),
            weight_two.grad[:, local_hidden_start:local_hidden_end].float(),
        ),
        (model.bias_two.grad.float(), bias_two.grad.float()),
    )
    relative_l2_error = max(
        float(
            torch.linalg.vector_norm(actual - expected).item()
            / max(torch.linalg.vector_norm(expected).item(), 1e-12)
        )
        for actual, expected in comparisons
    )
    error_names = tuple(errors)
    error_tensor = torch.tensor(
        [errors[name] for name in error_names],
        dtype=torch.float64,
        device=context.device,
    )
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    relative_error_tensor = torch.tensor(
        [relative_l2_error],
        dtype=torch.float64,
        device=context.device,
    )
    dist.all_reduce(relative_error_tensor, op=dist.ReduceOp.MAX)
    model.zero_grad(set_to_none=True)
    global_errors = {
        name: float(error_tensor[index].item()) for index, name in enumerate(error_names)
    }
    max_error = max(global_errors.values())
    tolerance_error = float(relative_error_tensor.item()) if autocast_dtype else max_error
    tolerance = args.rtol if autocast_dtype is not None else args.atol
    status = "ok" if tolerance_error <= tolerance else "failed"
    if not context.is_main:
        return {
            "max_error": max_error,
            "relative_l2_error": float(relative_error_tensor.item()),
            "status": status,
        }
    return {
        "max_error": max_error,
        "relative_l2_error": float(relative_error_tensor.item()),
        "tolerance": tolerance,
        "tolerance_mode": "relative_l2" if autocast_dtype is not None else "absolute",
        **global_errors,
        "status": status,
    }


def tp_dp_train(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        groups = _build_parallel_groups(context, args.tp_size)
        if args.hidden_features % groups.tensor_size != 0:
            raise ValueError("hidden-features 必须能被 tp-size 整除")
        if args.precision == "bf16":
            if context.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise ValueError("TP/DP BF16 需要支持 BF16 的 CUDA 设备")
            autocast_dtype: torch.dtype | None = torch.bfloat16
        else:
            autocast_dtype = None
        weights = _reference_parameters(
            args.in_features,
            args.hidden_features,
            args.out_features,
            args.seed,
        )
        model = TensorParallelMLP(
            weights,
            tensor_rank=groups.tensor_rank,
            tensor_size=groups.tensor_size,
            device=context.device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        probe = _correctness_probe(model, groups, context, weights, args, autocast_dtype)
        if probe["status"] != "ok":
            raise RuntimeError(f"TP/DP correctness probe failed: {probe}")
        context.barrier()

        measured: list[float] = []
        final_loss = 0.0
        for step in range(args.warmup_steps + args.steps):
            inputs, targets = _batch_for_step(
                args.batch_size,
                args.seq_length,
                args.in_features,
                args.out_features,
                args.seed,
                groups.data_rank,
                step + 1,
                context.device,
            )
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, groups.tensor, autocast_dtype)
            loss = F.mse_loss(output.float(), targets.float())
            (loss / groups.tensor_size).backward()
            _sync_tensor_replicated_gradients(model, groups)
            _sync_data_gradients(model, groups)
            optimizer.step()
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            elapsed = time.perf_counter() - started
            synchronized_elapsed = _reduce_max(elapsed, context)
            final_loss = float(loss.detach().item())
            if step >= args.warmup_steps and context.is_main:
                measured.append(synchronized_elapsed * 1000)

        if groups.tensor_rank == 0:
            loss_tensor = torch.tensor(final_loss, dtype=torch.float64, device=context.device)
            dist.reduce(loss_tensor, dst=0, group=groups.data)
            if context.is_main:
                final_loss = float(loss_tensor.item() / groups.data_size)

        if not context.is_main:
            return None
        mean_step_ms = statistics.fmean(measured)
        global_tokens = args.batch_size * args.seq_length * groups.data_size
        payload = {
            "benchmark": "tp_dp_training",
            "status": "ok",
            "strategy": "tp_dp",
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "precision": args.precision,
            "world_size": context.world_size,
            "tp_degree": groups.tensor_size,
            "dp_degree": groups.data_size,
            "batch_size_per_rank": args.batch_size,
            "seq_length": args.seq_length,
            "in_features": args.in_features,
            "hidden_features": args.hidden_features,
            "out_features": args.out_features,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "loss": final_loss,
            "step_time_ms": mean_step_ms,
            "step_time_p50_ms": float(torch.tensor(measured).quantile(0.5).item()),
            "step_time_p95_ms": float(torch.tensor(measured).quantile(0.95).item()),
            "tokens_per_sec": global_tokens / (mean_step_ms / 1000),
            "communication_bytes_per_step_estimate": int(
                (
                    args.batch_size
                    * args.seq_length
                    * args.out_features
                    * (2 if args.precision == "bf16" else 4)
                    * 2
                    * max(groups.tensor_size - 1, 0)
                )
                + (
                    args.out_features
                    * 4
                    * 2
                    * max(groups.tensor_size - 1, 0)
                    / groups.tensor_size
                )
                + (
                    sum(parameter.numel() for parameter in model.parameters())
                    * 4
                    * 2
                    * max(groups.data_size - 1, 0)
                    / groups.data_size
                )
            ),
            "correctness": probe,
            "parallel_groups": {
                "tensor_rank_layout": "rank // tp_size = dp_rank; rank % tp_size = tp_rank",
                "tensor_size": groups.tensor_size,
                "data_size": groups.data_size,
            },
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()
