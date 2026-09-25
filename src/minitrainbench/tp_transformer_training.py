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
from .tp_training import (
    ParallelGroups,
    _all_reduce_sum,
    _build_parallel_groups,
    _reduce_max,
    _sync_data_gradients,
)


@dataclass(frozen=True)
class TransformerWeights:
    ln1_weight: torch.Tensor
    ln1_bias: torch.Tensor
    q_weight: torch.Tensor
    q_bias: torch.Tensor
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor
    out_weight: torch.Tensor
    out_bias: torch.Tensor
    ln2_weight: torch.Tensor
    ln2_bias: torch.Tensor
    up_weight: torch.Tensor
    up_bias: torch.Tensor
    down_weight: torch.Tensor
    down_bias: torch.Tensor


def validate_transformer_shape(
    d_model: int,
    n_heads: int,
    mlp_hidden_size: int,
    tp_size: int,
) -> None:
    if d_model < 1 or n_heads < 1 or mlp_hidden_size < 1 or tp_size < 1:
        raise ValueError("d-model、n-heads、mlp-hidden-size 和 tp-size 必须为正数")
    if d_model % n_heads != 0:
        raise ValueError("d-model 必须能被 n-heads 整除")
    if d_model % tp_size != 0:
        raise ValueError("d-model 必须能被 tp-size 整除")
    if n_heads % tp_size != 0:
        raise ValueError("n-heads 必须能被 tp-size 整除")
    if mlp_hidden_size % tp_size != 0:
        raise ValueError("mlp-hidden-size 必须能被 tp-size 整除")


def validate_transformer_world_size(world_size: int, tp_size: int) -> None:
    if tp_size < 1 or world_size < 1 or world_size % tp_size != 0:
        raise ValueError(
            f"tp-size 必须是 world size 的正整数因子，当前 tp-size={tp_size}、"
            f"world-size={world_size}"
        )


def _make_reference_weights(
    d_model: int,
    mlp_hidden_size: int,
    seed: int,
) -> TransformerWeights:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = d_model**-0.5

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    def zeros(size: int) -> torch.Tensor:
        return torch.zeros(size, dtype=torch.float32)

    return TransformerWeights(
        ln1_weight=torch.ones(d_model),
        ln1_bias=zeros(d_model),
        q_weight=random((d_model, d_model)),
        q_bias=zeros(d_model),
        k_weight=random((d_model, d_model)),
        k_bias=zeros(d_model),
        v_weight=random((d_model, d_model)),
        v_bias=zeros(d_model),
        out_weight=random((d_model, d_model)),
        out_bias=zeros(d_model),
        ln2_weight=torch.ones(d_model),
        ln2_bias=zeros(d_model),
        up_weight=random((mlp_hidden_size, d_model)),
        up_bias=zeros(mlp_hidden_size),
        down_weight=random((d_model, mlp_hidden_size)),
        down_bias=zeros(d_model),
    )


def _transformer_batch_for_step(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    seed: int,
    data_rank: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + data_rank * 100_003 + step * 1_000_003
    )
    shape = (batch_size, sequence_length, d_model)
    inputs = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
    targets = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
    return inputs, targets


class TPColumnLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if weight.shape[0] % tensor_size != 0:
            raise ValueError("column parallel output size 必须能被 tp-size 整除")
        local_size = weight.shape[0] // tensor_size
        start = tensor_rank * local_size
        end = start + local_size
        self.weight = nn.Parameter(weight[start:end].to(device).clone())
        self.bias = nn.Parameter(bias[start:end].to(device).clone())
        self.local_size = local_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class TPColumnQKV(nn.Module):
    def __init__(
        self,
        weights: TransformerWeights,
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.q = TPColumnLinear(
            weights.q_weight,
            weights.q_bias,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.k = TPColumnLinear(
            weights.k_weight,
            weights.k_bias,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.v = TPColumnLinear(
            weights.v_weight,
            weights.v_bias,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.q(inputs), self.k(inputs), self.v(inputs)


class TPRowLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if weight.shape[1] % tensor_size != 0:
            raise ValueError("row parallel input size 必须能被 tp-size 整除")
        local_size = weight.shape[1] // tensor_size
        start = tensor_rank * local_size
        end = start + local_size
        self.weight = nn.Parameter(weight[:, start:end].to(device).clone())
        self.local_size = local_size

    def forward(
        self,
        inputs: torch.Tensor,
        tensor_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        partial = F.linear(inputs, self.weight, None)
        return _all_reduce_sum(partial, tensor_group)


class TPTransformerBlock(nn.Module):
    def __init__(
        self,
        weights: TransformerWeights,
        *,
        d_model: int,
        n_heads: int,
        mlp_hidden_size: int,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, device=device)
        self.ln2 = nn.LayerNorm(d_model, device=device)
        with torch.no_grad():
            self.ln1.weight.copy_(weights.ln1_weight.to(device))
            self.ln1.bias.copy_(weights.ln1_bias.to(device))
            self.ln2.weight.copy_(weights.ln2_weight.to(device))
            self.ln2.bias.copy_(weights.ln2_bias.to(device))
        self.qkv = TPColumnQKV(
            weights,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.attention_out = TPRowLinear(
            weights.out_weight,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.attention_bias = nn.Parameter(weights.out_bias.to(device).clone())
        self.up = TPColumnLinear(
            weights.up_weight,
            weights.up_bias,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.down = TPRowLinear(
            weights.down_weight,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.down_bias = nn.Parameter(weights.down_bias.to(device).clone())
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_heads = n_heads // tensor_size

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
            normalized = self.ln1(inputs)
            query, key, value = self.qkv(normalized)
            batch, sequence, _ = query.shape
            query = query.reshape(batch, sequence, self.local_heads, self.head_dim).transpose(1, 2)
            key = key.reshape(batch, sequence, self.local_heads, self.head_dim).transpose(1, 2)
            value = value.reshape(batch, sequence, self.local_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim**-0.5
            causal_mask = torch.triu(
                torch.ones(sequence, sequence, dtype=torch.bool, device=inputs.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probabilities, value)
            attended = attended.transpose(1, 2).reshape(
                batch,
                sequence,
                self.local_heads * self.head_dim,
            )
            residual = inputs + self.attention_out(attended, tensor_group) + self.attention_bias
            normalized = self.ln2(residual)
            hidden = F.gelu(self.up(normalized))
            output = self.down(hidden, tensor_group) + self.down_bias
            return residual + output


def _reference_transformer_forward_complete(
    inputs: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    d_model: int,
    n_heads: int,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    with torch.autocast(
        device_type=inputs.device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        normalized = F.layer_norm(
            inputs,
            (d_model,),
            weights["ln1_weight"],
            weights["ln1_bias"],
            1e-5,
        )
        query = F.linear(normalized, weights["q_weight"], weights["q_bias"])
        key = F.linear(normalized, weights["k_weight"], weights["k_bias"])
        value = F.linear(normalized, weights["v_weight"], weights["v_bias"])
        batch, sequence, _ = query.shape
        head_dim = d_model // n_heads
        query = query.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * head_dim**-0.5
        causal_mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=inputs.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(batch, sequence, d_model)
        residual = inputs + F.linear(attended, weights["out_weight"], weights["out_bias"])
        normalized = F.layer_norm(
            residual,
            (d_model,),
            weights["ln2_weight"],
            weights["ln2_bias"],
            1e-5,
        )
        hidden = F.gelu(F.linear(normalized, weights["up_weight"], weights["up_bias"]))
        return residual + F.linear(hidden, weights["down_weight"], weights["down_bias"])


def _reference_weight_dict(
    weights: TransformerWeights,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device).detach().requires_grad_(True)
        for name, value in weights.__dict__.items()
    }


def _replicated_parameters(model: TPTransformerBlock) -> tuple[nn.Parameter, ...]:
    return (
        model.ln1.weight,
        model.ln1.bias,
        model.ln2.weight,
        model.ln2.bias,
        model.attention_bias,
        model.down_bias,
    )


def _sync_transformer_gradients(
    model: TPTransformerBlock,
    groups: ParallelGroups,
) -> None:
    if groups.tensor_size > 1:
        for parameter in _replicated_parameters(model):
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=groups.tensor)
    _sync_data_gradients(model, groups)


def _dp_batches_are_distinct(inputs: torch.Tensor, groups: ParallelGroups) -> bool:
    if groups.data_size <= 1:
        return True
    gathered = [torch.empty_like(inputs) for _ in range(groups.data_size)]
    dist.all_gather(gathered, inputs, group=groups.data)
    return all(not torch.equal(gathered[0], other) for other in gathered[1:])


def _dp_parameters_are_consistent(model: TPTransformerBlock, groups: ParallelGroups) -> bool:
    if groups.data_size <= 1:
        return True
    flattened = torch.cat(
        [parameter.detach().float().reshape(-1) for parameter in model.parameters()]
    )
    minimum = flattened.clone()
    maximum = flattened.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=groups.data)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=groups.data)
    return bool(torch.equal(minimum, maximum))


def _local_reference_gradient(
    name: str,
    gradient: torch.Tensor,
    model: TPTransformerBlock,
    tensor_rank: int,
) -> torch.Tensor:
    if name in {"q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias"}:
        local_size = model.qkv.q.local_size
        start = tensor_rank * local_size
        end = start + local_size
        return gradient[start:end]
    if name in {"up_weight", "up_bias"}:
        local_size = model.up.local_size
        start = tensor_rank * local_size
        end = start + local_size
        return gradient[start:end]
    if name in {"out_weight", "down_weight"}:
        local_size = model.qkv.q.local_size if name == "out_weight" else model.up.local_size
        start = tensor_rank * local_size
        end = start + local_size
        return gradient[:, start:end]
    return gradient


def _model_gradient_pairs(
    model: TPTransformerBlock,
    reference: dict[str, torch.Tensor],
    groups: ParallelGroups,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    actual = dict(model.named_parameters())
    mapping = (
        ("ln1_weight", "ln1.weight"),
        ("ln1_bias", "ln1.bias"),
        ("q_weight", "qkv.q.weight"),
        ("q_bias", "qkv.q.bias"),
        ("k_weight", "qkv.k.weight"),
        ("k_bias", "qkv.k.bias"),
        ("v_weight", "qkv.v.weight"),
        ("v_bias", "qkv.v.bias"),
        ("out_weight", "attention_out.weight"),
        ("out_bias", "attention_bias"),
        ("ln2_weight", "ln2.weight"),
        ("ln2_bias", "ln2.bias"),
        ("up_weight", "up.weight"),
        ("up_bias", "up.bias"),
        ("down_weight", "down.weight"),
        ("down_bias", "down_bias"),
    )
    pairs = []
    for reference_name, model_name in mapping:
        parameter = actual[model_name]
        assert parameter.grad is not None
        reference_gradient = reference[reference_name].grad
        assert reference_gradient is not None
        expected = _local_reference_gradient(
            reference_name,
            reference_gradient,
            model,
            groups.tensor_rank,
        )
        pairs.append((reference_name, parameter.grad, expected))
    return pairs


def _correctness_probe(
    model: TPTransformerBlock,
    groups: ParallelGroups,
    context: DistributedContext,
    weights: TransformerWeights,
    args: Any,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    inputs, targets = _transformer_batch_for_step(
        args.batch_size,
        args.seq_length,
        args.d_model,
        args.seed,
        groups.data_rank,
        0,
        context.device,
    )
    distinct_dp_batches = _dp_batches_are_distinct(inputs, groups)
    reference = _reference_weight_dict(weights, context.device)
    model.zero_grad(set_to_none=True)
    output = model(inputs, groups.tensor, autocast_dtype)
    reference_output = _reference_transformer_forward_complete(
        inputs,
        reference,
        d_model=args.d_model,
        n_heads=args.n_heads,
        autocast_dtype=autocast_dtype,
    )
    loss = F.mse_loss(output.float(), targets.float())
    reference_loss = F.mse_loss(reference_output.float(), targets.float())
    (loss / groups.tensor_size).backward()
    _sync_transformer_gradients(model, groups)
    reference_loss.backward()
    for parameter in reference.values():
        if groups.data_size > 1:
            dist.all_reduce(parameter.grad, group=groups.data)
            parameter.grad.div_(groups.data_size)

    errors: dict[str, float] = {
        "forward_max_error": float((output.float() - reference_output.float()).abs().max().item()),
        "loss_error": float(abs(loss.item() - reference_loss.item())),
    }
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("forward", output.float(), reference_output.float()),
        ("loss", loss.detach().reshape(1), reference_loss.detach().reshape(1)),
    ]
    for name, actual, expected in _model_gradient_pairs(model, reference, groups):
        error = float((actual.float() - expected.float()).abs().max().item())
        errors[f"{name}_grad_error"] = error
        comparisons.append((name, actual.float(), expected.float()))
    relative_l2_by_name = {
        name: float(torch.linalg.vector_norm(actual - expected).item())
        / max(float(torch.linalg.vector_norm(expected).item()), 1e-2)
        for name, actual, expected in comparisons
    }
    relative_l2_error = max(relative_l2_by_name.values())
    error_names = tuple(errors)
    error_tensor = torch.tensor(
        [errors[name] for name in error_names], dtype=torch.float64, device=context.device
    )
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    relative_tensor = torch.tensor([relative_l2_error], dtype=torch.float64, device=context.device)
    dist.all_reduce(relative_tensor, op=dist.ReduceOp.MAX)
    model.zero_grad(set_to_none=True)
    max_error = float(error_tensor.max().item())
    relative_error = float(relative_tensor.item())
    tolerance = args.rtol if autocast_dtype is not None else args.atol
    tolerance_error = relative_error if autocast_dtype is not None else max_error
    status = "ok" if tolerance_error <= tolerance and distinct_dp_batches else "failed"
    result: dict[str, Any] = {
        "max_error": max_error,
        "relative_l2_error": relative_error,
        "relative_l2_floor": 1e-2,
        "relative_l2_by_tensor": relative_l2_by_name,
        "data_parallel_batches_distinct": distinct_dp_batches,
        "tolerance": tolerance,
        "tolerance_mode": "relative_l2" if autocast_dtype is not None else "absolute",
        **{name: float(error_tensor[index].item()) for index, name in enumerate(error_names)},
        "status": status,
    }
    return result if context.is_main else {"status": status}


def _communication_estimate(
    model: TPTransformerBlock,
    groups: ParallelGroups,
    args: Any,
) -> dict[str, int]:
    element_size = 2 if args.precision == "bf16" else 4
    activation_bytes = args.batch_size * args.seq_length * args.d_model * element_size
    tp_activation = 2 * activation_bytes * max(groups.tensor_size - 1, 0)
    replicated_parameters = sum(parameter.numel() for parameter in _replicated_parameters(model))
    tp_replicated_gradients = (
        replicated_parameters
        * element_size
        * 2
        * max(groups.tensor_size - 1, 0)
        / max(groups.tensor_size, 1)
    )
    dp_gradients = (
        sum(parameter.numel() for parameter in model.parameters())
        * element_size
        * 2
        * max(groups.data_size - 1, 0)
        / max(groups.data_size, 1)
    )
    return {
        "tp_bytes_estimate": int(tp_activation + tp_replicated_gradients),
        "dp_bytes_estimate": int(dp_gradients),
        "total_bytes_estimate": int(tp_activation + tp_replicated_gradients + dp_gradients),
    }


def tp_dp_transformer_train(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        validate_transformer_world_size(context.world_size, args.tp_size)
        groups = _build_parallel_groups(context, args.tp_size)
        mlp_hidden_size = args.mlp_hidden_size or 4 * args.d_model
        validate_transformer_shape(
            args.d_model,
            args.n_heads,
            mlp_hidden_size,
            groups.tensor_size,
        )
        if args.precision == "bf16":
            if context.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise ValueError("TP/DP BF16 需要支持 BF16 的 CUDA 设备")
            autocast_dtype: torch.dtype | None = torch.bfloat16
        else:
            autocast_dtype = None
        weights = _make_reference_weights(args.d_model, mlp_hidden_size, args.seed)
        model = TPTransformerBlock(
            weights,
            d_model=args.d_model,
            n_heads=args.n_heads,
            mlp_hidden_size=mlp_hidden_size,
            tensor_rank=groups.tensor_rank,
            tensor_size=groups.tensor_size,
            device=context.device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        probe = _correctness_probe(model, groups, context, weights, args, autocast_dtype)
        if probe.get("status") != "ok":
            raise RuntimeError(f"Transformer TP/DP correctness probe failed: {probe}")
        context.barrier()

        measured: list[float] = []
        final_loss = 0.0
        for step in range(args.warmup_steps + args.steps):
            inputs, targets = _transformer_batch_for_step(
                args.batch_size,
                args.seq_length,
                args.d_model,
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
            _sync_transformer_gradients(model, groups)
            optimizer.step()
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            elapsed = _reduce_max(time.perf_counter() - started, context)
            final_loss = float(loss.detach().item())
            if step >= args.warmup_steps and context.is_main:
                measured.append(elapsed * 1000)

        parameters_consistent = _dp_parameters_are_consistent(model, groups)
        if not parameters_consistent:
            raise RuntimeError("optimizer step produced divergent DP replica parameters")
        probe["dp_parameters_consistent_after_step"] = parameters_consistent

        if groups.tensor_rank == 0:
            loss_tensor = torch.tensor(final_loss, dtype=torch.float64, device=context.device)
            dist.reduce(loss_tensor, dst=0, group=groups.data)
            if context.is_main:
                final_loss = float(loss_tensor.item() / groups.data_size)

        if not context.is_main:
            return None
        mean_step_ms = statistics.fmean(measured)
        communication = _communication_estimate(model, groups, args)
        global_tokens = args.batch_size * args.seq_length * groups.data_size
        payload = {
            "benchmark": "tp_dp_transformer_training",
            "benchmark_schema_version": 1,
            "status": "ok",
            "strategy": "tp_dp_transformer",
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "precision": args.precision,
            "world_size": context.world_size,
            "tp_degree": groups.tensor_size,
            "dp_degree": groups.data_size,
            "batch_size_per_rank": args.batch_size,
            "seq_length": args.seq_length,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "mlp_hidden_size": mlp_hidden_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "loss": final_loss,
            "step_time_ms": mean_step_ms,
            "step_time_p50_ms": float(torch.tensor(measured).quantile(0.5).item()),
            "step_time_p95_ms": float(torch.tensor(measured).quantile(0.95).item()),
            "tokens_per_sec": global_tokens / (mean_step_ms / 1000),
            "communication_bytes_per_step_estimate": communication["total_bytes_estimate"],
            "tp_communication_bytes_estimate": communication["tp_bytes_estimate"],
            "dp_communication_bytes_estimate": communication["dp_bytes_estimate"],
            "communication_estimate_scope": "activation all-reduce plus gradient all-reduce",
            "correctness": probe,
            "parallel_groups": {
                "tensor_rank_layout": (
                    "tp_rank = global_rank % tp_size; "
                    "dp_rank = global_rank // tp_size"
                ),
                "tensor_size": groups.tensor_size,
                "data_size": groups.data_size,
            },
            "limitations": [
                "MHA only; no GQA/MQA",
                "no checkpoint/resume or optimizer state sharding",
                "no pipeline parallel or multi-node TP",
                "regression target instead of vocab-parallel cross entropy",
            ],
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()
