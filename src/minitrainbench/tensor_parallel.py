from __future__ import annotations

import json
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .distributed import setup_distributed
from .runtime import _write_json


class ColumnParallelLinear(nn.Module):
    """按输出维切分 Linear，模拟 Megatron 的 ColumnParallelLinear。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if out_features % world_size != 0:
            raise ValueError("out_features 必须能被 tensor parallel degree 整除")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.local_out_features = out_features // world_size
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features, device=device)
        )
        self.bias = nn.Parameter(torch.empty(self.local_out_features, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """按输入维切分 Linear，模拟 Megatron 的 RowParallelLinear。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if in_features % world_size != 0:
            raise ValueError("in_features 必须能被 tensor parallel degree 整除")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.local_in_features = in_features // world_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features, device=device)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.weight, None)


def _make_reference_tensors(args: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.batch_size,
        args.seq_length,
        args.in_features,
        generator=generator,
    ).to(device)
    weight = torch.randn(
        args.out_features,
        args.in_features,
        generator=generator,
    ).to(device)
    bias = torch.randn(args.out_features, generator=generator).to(device)
    grad_output = torch.randn(
        args.batch_size,
        args.seq_length,
        args.out_features,
        generator=generator,
    ).to(device)
    return inputs, weight, bias, grad_output


def _slice_range(total: int, world_size: int, rank: int) -> tuple[int, int]:
    local = total // world_size
    start = rank * local
    return start, start + local


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _reduce_errors(
    errors: dict[str, float],
    context: Any,
) -> dict[str, float] | None:
    keys = sorted(errors)
    values = torch.tensor(
        [errors[key] for key in keys],
        dtype=torch.float64,
        device=context.device,
    )
    dist.reduce(values, dst=0, op=dist.ReduceOp.MAX)
    if not context.is_main:
        return None
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def _reference_backward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_bias = bias.detach().clone().requires_grad_(True)
    reference_output = nn.functional.linear(
        reference_inputs,
        reference_weight,
        reference_bias,
    )
    reference_output.backward(grad_output)
    assert reference_inputs.grad is not None
    assert reference_weight.grad is not None
    assert reference_bias.grad is not None
    return (
        reference_output.detach(),
        reference_inputs.grad.detach(),
        reference_weight.grad.detach(),
        reference_bias.grad.detach(),
    )


def _check_column_parallel(
    args: Any,
    context: Any,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float] | None:
    reference_output, reference_input_grad, reference_weight_grad, reference_bias_grad = (
        reference
    )
    start, end = _slice_range(args.out_features, context.world_size, context.rank)
    module = ColumnParallelLinear(
        args.in_features,
        args.out_features,
        rank=context.rank,
        world_size=context.world_size,
        device=context.device,
    )
    with torch.no_grad():
        module.weight.copy_(weight[start:end])
        module.bias.copy_(bias[start:end])

    local_inputs = inputs.detach().clone().requires_grad_(True)
    local_output = module(local_inputs)
    gathered = [torch.empty_like(local_output) for _ in range(context.world_size)]
    dist.all_gather(gathered, local_output.detach())
    full_output = torch.cat(gathered, dim=-1)
    local_output.backward(grad_output[..., start:end].contiguous())
    assert local_inputs.grad is not None
    assert module.weight.grad is not None
    assert module.bias.grad is not None

    input_grad = local_inputs.grad.detach().clone()
    dist.all_reduce(input_grad, op=dist.ReduceOp.SUM)
    local_errors = {
        "column_forward": _max_abs(full_output, reference_output),
        "column_input_grad": _max_abs(input_grad, reference_input_grad),
        "column_weight_grad": _max_abs(module.weight.grad, reference_weight_grad[start:end]),
        "column_bias_grad": _max_abs(module.bias.grad, reference_bias_grad[start:end]),
    }
    return _reduce_errors(local_errors, context)


def _check_row_parallel(
    args: Any,
    context: Any,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float] | None:
    reference_output, reference_input_grad, reference_weight_grad, reference_bias_grad = (
        reference
    )
    start, end = _slice_range(args.in_features, context.world_size, context.rank)
    module = RowParallelLinear(
        args.in_features,
        args.out_features,
        rank=context.rank,
        world_size=context.world_size,
        device=context.device,
    )
    with torch.no_grad():
        module.weight.copy_(weight[:, start:end])

    local_inputs = inputs[..., start:end].detach().clone().requires_grad_(True)
    local_partial = module(local_inputs)
    full_output = local_partial.detach().clone()
    dist.all_reduce(full_output, op=dist.ReduceOp.SUM)
    full_output = full_output + bias
    local_partial.backward(grad_output)
    assert local_inputs.grad is not None
    assert module.weight.grad is not None

    expected_bias_grad = grad_output.sum(dim=(0, 1))
    local_errors = {
        "row_forward": _max_abs(full_output, reference_output),
        "row_input_grad": _max_abs(
            local_inputs.grad,
            reference_input_grad[..., start:end],
        ),
        "row_weight_grad": _max_abs(module.weight.grad, reference_weight_grad[:, start:end]),
        "row_bias_grad": _max_abs(expected_bias_grad, reference_bias_grad),
    }
    return _reduce_errors(local_errors, context)


def tensor_parallel_check(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        if context.world_size < 2:
            raise ValueError("tensor parallel check 至少需要 2 个进程")
        if args.in_features % context.world_size != 0:
            raise ValueError("in-features 必须能被 tensor parallel degree 整除")
        if args.out_features % context.world_size != 0:
            raise ValueError("out-features 必须能被 tensor parallel degree 整除")

        tensors = _make_reference_tensors(args, context.device)
        reference = _reference_backward(*tensors)
        column_errors = _check_column_parallel(args, context, *tensors, reference)
        row_errors = _check_row_parallel(args, context, *tensors, reference)

        if not context.is_main:
            return None

        assert column_errors is not None
        assert row_errors is not None
        all_errors = {**column_errors, **row_errors}
        forward_max_error = max(
            all_errors["column_forward"],
            all_errors["row_forward"],
        )
        grad_max_error = max(
            value
            for key, value in all_errors.items()
            if key.endswith("_grad")
        )
        status = "ok" if max(forward_max_error, grad_max_error) <= args.atol else "failed"
        payload = {
            "benchmark": "tensor_parallel",
            "status": status,
            "tp_degree": context.world_size,
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "seed": args.seed,
            "atol": args.atol,
            "forward_max_error": forward_max_error,
            "grad_max_error": grad_max_error,
            "errors": all_errors,
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()
