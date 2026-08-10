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


def _make_mlp_tensors(args: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.batch_size,
        args.seq_length,
        args.in_features,
        generator=generator,
    ).to(device)
    w1 = torch.randn(args.hidden_features, args.in_features, generator=generator).to(device)
    b1 = torch.randn(args.hidden_features, generator=generator).to(device)
    w2 = torch.randn(args.out_features, args.hidden_features, generator=generator).to(device)
    b2 = torch.randn(args.out_features, generator=generator).to(device)
    grad_output = torch.randn(
        args.batch_size,
        args.seq_length,
        args.out_features,
        generator=generator,
    ).to(device)
    return inputs, w1, b1, w2, b2, grad_output


def _reference_mlp_backward(
    inputs: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_w1 = w1.detach().clone().requires_grad_(True)
    reference_b1 = b1.detach().clone().requires_grad_(True)
    reference_w2 = w2.detach().clone().requires_grad_(True)
    reference_b2 = b2.detach().clone().requires_grad_(True)
    hidden = nn.functional.gelu(
        nn.functional.linear(reference_inputs, reference_w1, reference_b1)
    )
    output = nn.functional.linear(hidden, reference_w2, reference_b2)
    output.backward(grad_output)
    assert reference_inputs.grad is not None
    assert reference_w1.grad is not None
    assert reference_b1.grad is not None
    assert reference_w2.grad is not None
    assert reference_b2.grad is not None
    return (
        output.detach(),
        reference_inputs.grad.detach(),
        reference_w1.grad.detach(),
        reference_b1.grad.detach(),
        reference_w2.grad.detach(),
        reference_b2.grad.detach(),
    )


def tensor_parallel_mlp_check(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        if context.world_size < 2:
            raise ValueError("TP MLP correctness check 至少需要 2 个进程")
        if args.hidden_features % context.world_size != 0:
            raise ValueError("hidden-features 必须能被 tensor parallel degree 整除")
        tensors = _make_mlp_tensors(args, context.device)
        (
            reference_output,
            reference_input_grad,
            reference_w1_grad,
            reference_b1_grad,
            reference_w2_grad,
            reference_b2_grad,
        ) = _reference_mlp_backward(*tensors)
        inputs, w1, b1, w2, b2, grad_output = tensors
        start, end = _slice_range(args.hidden_features, context.world_size, context.rank)
        column = ColumnParallelLinear(
            args.in_features,
            args.hidden_features,
            rank=context.rank,
            world_size=context.world_size,
            device=context.device,
        )
        row = RowParallelLinear(
            args.hidden_features,
            args.out_features,
            rank=context.rank,
            world_size=context.world_size,
            device=context.device,
        )
        with torch.no_grad():
            column.weight.copy_(w1[start:end])
            column.bias.copy_(b1[start:end])
            row.weight.copy_(w2[:, start:end])

        local_inputs = inputs.detach().clone().requires_grad_(True)
        local_hidden = nn.functional.gelu(column(local_inputs))
        local_partial = row(local_hidden)
        full_output = local_partial.detach().clone()
        dist.all_reduce(full_output, op=dist.ReduceOp.SUM)
        full_output = full_output + b2
        local_partial.backward(grad_output)
        assert local_inputs.grad is not None
        assert column.weight.grad is not None
        assert column.bias.grad is not None
        assert row.weight.grad is not None

        input_grad = local_inputs.grad.detach().clone()
        dist.all_reduce(input_grad, op=dist.ReduceOp.SUM)
        local_errors = {
            "tp_mlp_forward": _max_abs(full_output, reference_output),
            "tp_mlp_input_grad": _max_abs(input_grad, reference_input_grad),
            "tp_mlp_w1_grad": _max_abs(column.weight.grad, reference_w1_grad[start:end]),
            "tp_mlp_b1_grad": _max_abs(column.bias.grad, reference_b1_grad[start:end]),
            "tp_mlp_w2_grad": _max_abs(row.weight.grad, reference_w2_grad[:, start:end]),
            "tp_mlp_b2_grad": _max_abs(grad_output.sum(dim=(0, 1)), reference_b2_grad),
        }
        reduced_errors = _reduce_errors(local_errors, context)
        if not context.is_main:
            return None
        assert reduced_errors is not None
        forward_max_error = reduced_errors["tp_mlp_forward"]
        grad_max_error = max(
            value
            for key, value in reduced_errors.items()
            if key.endswith("_grad")
        )
        status = "ok" if max(forward_max_error, grad_max_error) <= args.atol else "failed"
        communication_bytes = int(
            (
                args.batch_size * args.seq_length * args.out_features
                + args.batch_size * args.seq_length * args.in_features
            )
            * inputs.element_size()
        )
        payload = {
            "benchmark": "tensor_parallel_mlp",
            "status": status,
            "tp_degree": context.world_size,
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "in_features": args.in_features,
            "hidden_features": args.hidden_features,
            "out_features": args.out_features,
            "seed": args.seed,
            "atol": args.atol,
            "forward_max_error": forward_max_error,
            "grad_max_error": grad_max_error,
            "collective_count": 2,
            "communication_bytes": communication_bytes,
            "errors": reduced_errors,
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()


def _make_sequence_tensors(args: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        generator=generator,
    ).to(device)
    weight = torch.randn(args.hidden_size, generator=generator).to(device)
    bias = torch.randn(args.hidden_size, generator=generator).to(device)
    grad_output = torch.randn(
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        generator=generator,
    ).to(device)
    dropout_mask = (
        torch.rand(
            args.batch_size,
            args.seq_length,
            args.hidden_size,
            generator=generator,
        )
        >= args.dropout
    ).to(device)
    return inputs, weight, bias, grad_output, dropout_mask


def _dropout_scale(dropout: float) -> float:
    if dropout < 0 or dropout >= 1:
        raise ValueError("dropout 必须在 [0, 1) 范围内")
    return 1.0 / (1.0 - dropout) if dropout > 0 else 1.0


def tensor_parallel_sequence_check(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        if context.world_size < 2:
            raise ValueError("sequence parallel correctness check 至少需要 2 个进程")
        if args.seq_length % context.world_size != 0:
            raise ValueError("seq-length 必须能被 tensor parallel degree 整除")
        inputs, weight, bias, grad_output, dropout_mask = _make_sequence_tensors(
            args,
            context.device,
        )
        scale = _dropout_scale(args.dropout)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_weight = weight.detach().clone().requires_grad_(True)
        reference_bias = bias.detach().clone().requires_grad_(True)
        reference_output = nn.functional.layer_norm(
            reference_inputs,
            (args.hidden_size,),
            reference_weight,
            reference_bias,
        )
        reference_output = reference_output * dropout_mask * scale
        reference_output.backward(grad_output)
        assert reference_inputs.grad is not None
        assert reference_weight.grad is not None
        assert reference_bias.grad is not None

        start, end = _slice_range(args.seq_length, context.world_size, context.rank)
        local_inputs = inputs[:, start:end].detach().clone().requires_grad_(True)
        local_weight = weight.detach().clone().requires_grad_(True)
        local_bias = bias.detach().clone().requires_grad_(True)
        local_output = nn.functional.layer_norm(
            local_inputs,
            (args.hidden_size,),
            local_weight,
            local_bias,
        )
        local_output = local_output * dropout_mask[:, start:end] * scale
        gathered = [torch.empty_like(local_output) for _ in range(context.world_size)]
        dist.all_gather(gathered, local_output.detach())
        full_output = torch.cat(gathered, dim=1)
        local_output.backward(grad_output[:, start:end].contiguous())
        assert local_inputs.grad is not None
        assert local_weight.grad is not None
        assert local_bias.grad is not None
        weight_grad = local_weight.grad.detach().clone()
        bias_grad = local_bias.grad.detach().clone()
        dist.all_reduce(weight_grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(bias_grad, op=dist.ReduceOp.SUM)
        local_errors = {
            "sequence_forward": _max_abs(full_output, reference_output),
            "sequence_input_grad": _max_abs(
                local_inputs.grad,
                reference_inputs.grad[:, start:end],
            ),
            "sequence_weight_grad": _max_abs(weight_grad, reference_weight.grad),
            "sequence_bias_grad": _max_abs(bias_grad, reference_bias.grad),
        }
        reduced_errors = _reduce_errors(local_errors, context)
        if not context.is_main:
            return None
        assert reduced_errors is not None
        forward_max_error = reduced_errors["sequence_forward"]
        grad_max_error = max(
            value
            for key, value in reduced_errors.items()
            if key.endswith("_grad")
        )
        status = "ok" if max(forward_max_error, grad_max_error) <= args.atol else "failed"
        communication_bytes = int(
            (
                args.batch_size * args.seq_length * args.hidden_size
                + 2 * args.hidden_size
            )
            * inputs.element_size()
        )
        payload = {
            "benchmark": "sequence_parallel",
            "status": status,
            "tp_degree": context.world_size,
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "sequence_shard_size": args.seq_length // context.world_size,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "seed": args.seed,
            "atol": args.atol,
            "forward_max_error": forward_max_error,
            "grad_max_error": grad_max_error,
            "collective_count": 3,
            "communication_bytes": communication_bytes,
            "errors": reduced_errors,
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()


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
