from __future__ import annotations

import json
import math
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
from .tp_transformer_training import (
    TPTransformerBlock,
    TransformerWeights,
    _dp_parameters_are_consistent,
    _make_reference_weights,
    _replicated_parameters,
    validate_transformer_shape,
    validate_transformer_world_size,
)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        local_logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_start: int,
        vocab_size: int,
        tensor_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        if local_logits.shape[:-1] != targets.shape:
            raise ValueError("local logits 和 targets 的 shape 不匹配")
        local_vocab = local_logits.shape[-1]
        if local_vocab < 1:
            raise ValueError("local vocab size 必须为正数")
        logits = local_logits.float()
        target_local = targets - vocab_start
        target_mask = (targets < vocab_start) | (targets >= vocab_start + local_vocab)
        safe_target = target_local.clamp(min=0, max=local_vocab - 1)
        local_target_logit = logits.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
        local_target_logit = local_target_logit.masked_fill(target_mask, 0.0)
        global_max = logits.max(dim=-1).values
        local_exp_sum = torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1)
        if tensor_group is not None:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tensor_group)
            shifted = logits - global_max.unsqueeze(-1)
            local_exp_sum = torch.exp(shifted).sum(dim=-1)
            dist.all_reduce(local_exp_sum, op=dist.ReduceOp.SUM, group=tensor_group)
            dist.all_reduce(local_target_logit, op=dist.ReduceOp.SUM, group=tensor_group)
        else:
            shifted = logits - global_max.unsqueeze(-1)
        probabilities = torch.exp(shifted) / local_exp_sum.unsqueeze(-1)
        loss = torch.log(local_exp_sum) + global_max - local_target_logit
        context.save_for_backward(probabilities, target_mask, safe_target)
        context.input_dtype = local_logits.dtype
        context.tensor_size = dist.get_world_size(tensor_group) if tensor_group is not None else 1
        return loss.mean()

    @staticmethod
    def backward(
        context: Any,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        probabilities, target_mask, safe_target = context.saved_tensors
        gradient_logits = probabilities.clone()
        local_target = safe_target.unsqueeze(-1)
        target_gradient = torch.zeros_like(gradient_logits)
        target_gradient.scatter_(-1, local_target, 1.0)
        gradient_logits = torch.where(
            target_mask.unsqueeze(-1),
            gradient_logits,
            gradient_logits - target_gradient,
        )
        gradient_logits.div_(probabilities.shape[:-1].numel())
        gradient_logits.mul_(gradient)
        return gradient_logits.to(dtype=context.input_dtype), None, None, None, None, None


class _VocabParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        tensor_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        context.save_for_backward(hidden, weight)
        context.tensor_group = tensor_group
        return F.linear(hidden, weight.to(dtype=hidden.dtype), None)

    @staticmethod
    def backward(
        context: Any,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        hidden, weight = context.saved_tensors
        gradient_float = gradient.float()
        hidden_gradient = gradient_float.matmul(weight.float()).to(dtype=hidden.dtype)
        if context.tensor_group is not None:
            dist.all_reduce(hidden_gradient, op=dist.ReduceOp.SUM, group=context.tensor_group)
            hidden_gradient.div_(dist.get_world_size(context.tensor_group))
        weight_gradient = gradient_float.reshape(-1, gradient.shape[-1]).transpose(0, 1).matmul(
            hidden.float().reshape(-1, hidden.shape[-1])
        )
        return hidden_gradient, weight_gradient, None


class _ReplicatedIdentity(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        values: torch.Tensor,
        tensor_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        context.tensor_group = tensor_group
        return values

    @staticmethod
    def backward(
        context: Any,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        output = gradient.clone()
        if context.tensor_group is not None:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=context.tensor_group)
            output.div_(dist.get_world_size(context.tensor_group))
        return output, None


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_start: int,
    vocab_size: int,
    tensor_group: dist.ProcessGroup,
) -> torch.Tensor:
    tensor_size = dist.get_world_size(tensor_group) if tensor_group is not None else 1
    if vocab_size < 1 or local_logits.shape[-1] * tensor_size != vocab_size:
        raise ValueError("vocab-size 必须等于所有 TP rank 的 local vocab size 之和")
    return _VocabParallelCrossEntropy.apply(
        local_logits,
        targets,
        vocab_start,
        vocab_size,
        tensor_group,
    )


@dataclass(frozen=True)
class LanguageModelWeights:
    token_embedding: torch.Tensor
    position_embedding: torch.Tensor
    blocks: tuple[TransformerWeights, ...]
    final_norm_weight: torch.Tensor
    final_norm_bias: torch.Tensor
    lm_head: torch.Tensor


def _make_lm_weights(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    mlp_hidden_size: int,
    seed: int,
    max_position_embeddings: int = 2048,
) -> LanguageModelWeights:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = d_model**-0.5
    token_embedding = torch.randn(vocab_size, d_model, generator=generator) * scale
    position_embedding = torch.randn(
        max_position_embeddings,
        d_model,
        generator=generator,
    ) * scale
    blocks = tuple(
        _make_reference_weights(d_model, mlp_hidden_size, seed + 101 + layer_index)
        for layer_index in range(n_layers)
    )
    lm_head = torch.randn(vocab_size, d_model, generator=generator) * scale
    return LanguageModelWeights(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        blocks=blocks,
        final_norm_weight=torch.ones(d_model),
        final_norm_bias=torch.zeros(d_model),
        lm_head=lm_head,
    )


class TPVocabEmbedding(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if weight.shape[0] % tensor_size != 0:
            raise ValueError("vocab-size 必须能被 tp-size 整除")
        local_vocab = weight.shape[0] // tensor_size
        start = tensor_rank * local_vocab
        self.weight = nn.Parameter(weight[start : start + local_vocab].to(device).clone())
        self.vocab_start = start
        self.vocab_end = start + local_vocab
        self.vocab_size = weight.shape[0]

    def forward(self, token_ids: torch.Tensor, tensor_group: dist.ProcessGroup) -> torch.Tensor:
        local_ids = token_ids - self.vocab_start
        mask = (local_ids < 0) | (local_ids >= self.weight.shape[0])
        safe_ids = local_ids.clamp(min=0, max=self.weight.shape[0] - 1)
        local = F.embedding(safe_ids, self.weight)
        local = local.masked_fill(mask.unsqueeze(-1), 0.0)
        return _all_reduce_sum(local, tensor_group)


class TPVocabLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if weight.shape[0] % tensor_size != 0:
            raise ValueError("vocab-size 必须能被 tp-size 整除")
        local_vocab = weight.shape[0] // tensor_size
        start = tensor_rank * local_vocab
        self.weight = nn.Parameter(weight[start : start + local_vocab].to(device).clone())
        self.vocab_start = start
        self.vocab_end = start + local_vocab
        self.vocab_size = weight.shape[0]

    def forward(self, hidden: torch.Tensor, tensor_group: dist.ProcessGroup) -> torch.Tensor:
        return _VocabParallelLinear.apply(hidden, self.weight, tensor_group)


class TPTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_hidden_size: int,
        tensor_rank: int,
        tensor_size: int,
        device: torch.device,
        seed: int,
        weights: LanguageModelWeights | None = None,
    ) -> None:
        super().__init__()
        validate_lm_shape(vocab_size, d_model, n_heads, n_layers, mlp_hidden_size, tensor_size)
        self.weights = weights or _make_lm_weights(
            vocab_size,
            d_model,
            n_layers,
            mlp_hidden_size,
            seed,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.token_embedding = TPVocabEmbedding(
            self.weights.token_embedding,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )
        self.position_embedding = nn.Parameter(self.weights.position_embedding.to(device).clone())
        self.blocks = nn.ModuleList(
            TPTransformerBlock(
                block_weights,
                d_model=d_model,
                n_heads=n_heads,
                mlp_hidden_size=mlp_hidden_size,
                tensor_rank=tensor_rank,
                tensor_size=tensor_size,
                device=device,
            )
            for block_weights in self.weights.blocks
        )
        self.final_norm = nn.LayerNorm(d_model, device=device)
        with torch.no_grad():
            self.final_norm.weight.copy_(self.weights.final_norm_weight.to(device))
            self.final_norm.bias.copy_(self.weights.final_norm_bias.to(device))
        self.lm_head = TPVocabLinear(
            self.weights.lm_head,
            tensor_rank=tensor_rank,
            tensor_size=tensor_size,
            device=device,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        tensor_group: dist.ProcessGroup,
        autocast_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token ids 必须是 [batch, sequence]")
        if token_ids.shape[1] > self.position_embedding.shape[0]:
            raise ValueError("sequence length 超过 position embedding capacity")
        hidden = self.token_embedding(token_ids, tensor_group)
        hidden = hidden + self.position_embedding[: token_ids.shape[1]].unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden, tensor_group, autocast_dtype)
            hidden = _ReplicatedIdentity.apply(hidden, tensor_group)
        with torch.autocast(
            device_type=hidden.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            hidden = self.final_norm(hidden)
            return self.lm_head(hidden, tensor_group)


class ReferenceTransformerBlock(nn.Module):
    def __init__(self, weights: TransformerWeights, device: torch.device) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(weights.ln1_weight.shape[0], device=device)
        self.ln2 = nn.LayerNorm(weights.ln1_weight.shape[0], device=device)
        self.q_weight = nn.Parameter(weights.q_weight.to(device).clone())
        self.q_bias = nn.Parameter(weights.q_bias.to(device).clone())
        self.k_weight = nn.Parameter(weights.k_weight.to(device).clone())
        self.k_bias = nn.Parameter(weights.k_bias.to(device).clone())
        self.v_weight = nn.Parameter(weights.v_weight.to(device).clone())
        self.v_bias = nn.Parameter(weights.v_bias.to(device).clone())
        self.out_weight = nn.Parameter(weights.out_weight.to(device).clone())
        self.out_bias = nn.Parameter(weights.out_bias.to(device).clone())
        self.up_weight = nn.Parameter(weights.up_weight.to(device).clone())
        self.up_bias = nn.Parameter(weights.up_bias.to(device).clone())
        self.down_weight = nn.Parameter(weights.down_weight.to(device).clone())
        self.down_bias = nn.Parameter(weights.down_bias.to(device).clone())
        with torch.no_grad():
            self.ln1.weight.copy_(weights.ln1_weight.to(device))
            self.ln1.bias.copy_(weights.ln1_bias.to(device))
            self.ln2.weight.copy_(weights.ln2_weight.to(device))
            self.ln2.bias.copy_(weights.ln2_bias.to(device))

    def forward(self, inputs: torch.Tensor, n_heads: int, autocast_dtype: torch.dtype | None) -> torch.Tensor:
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            normalized = self.ln1(inputs)
            query = F.linear(normalized, self.q_weight, self.q_bias)
            key = F.linear(normalized, self.k_weight, self.k_bias)
            value = F.linear(normalized, self.v_weight, self.v_bias)
            batch, sequence, d_model = query.shape
            head_dim = d_model // n_heads
            query = query.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
            key = key.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
            value = value.reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * head_dim**-0.5
            causal_mask = torch.triu(
                torch.ones(sequence, sequence, dtype=torch.bool, device=inputs.device),
                diagonal=1,
            )
            attended = torch.matmul(torch.softmax(scores.masked_fill(causal_mask, float("-inf")), dim=-1), value)
            attended = attended.transpose(1, 2).reshape(batch, sequence, d_model)
            residual = inputs + F.linear(attended, self.out_weight, self.out_bias)
            normalized = self.ln2(residual)
            hidden = F.gelu(F.linear(normalized, self.up_weight, self.up_bias))
            return residual + F.linear(hidden, self.down_weight, self.down_bias)


class ReferenceTransformerLM(nn.Module):
    def __init__(
        self,
        weights: LanguageModelWeights,
        d_model: int,
        n_heads: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.token_embedding = nn.Embedding.from_pretrained(weights.token_embedding.to(device).clone(), freeze=False)
        self.position_embedding = nn.Parameter(weights.position_embedding.to(device).clone())
        self.blocks = nn.ModuleList(
            ReferenceTransformerBlock(block_weights, device) for block_weights in weights.blocks
        )
        self.final_norm = nn.LayerNorm(d_model, device=device)
        with torch.no_grad():
            self.final_norm.weight.copy_(weights.final_norm_weight.to(device))
            self.final_norm.bias.copy_(weights.final_norm_bias.to(device))
        self.lm_head = nn.Linear(d_model, weights.lm_head.shape[0], bias=False, device=device)
        with torch.no_grad():
            self.lm_head.weight.copy_(weights.lm_head.to(device))

    def forward(self, token_ids: torch.Tensor, autocast_dtype: torch.dtype | None) -> torch.Tensor:
        hidden = F.embedding(token_ids, self.token_embedding.weight)
        hidden = hidden + self.position_embedding[: token_ids.shape[1]].unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden, self.n_heads, autocast_dtype)
        with torch.autocast(
            device_type=hidden.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            return self.lm_head(self.final_norm(hidden))


def validate_lm_shape(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mlp_hidden_size: int,
    tp_size: int,
) -> None:
    if vocab_size < 1 or n_layers < 1 or tp_size < 1:
        raise ValueError("vocab-size、n-layers 和 tp-size 必须为正数")
    if vocab_size % tp_size != 0:
        raise ValueError("vocab-size 必须能被 tp-size 整除")
    validate_transformer_shape(d_model, n_heads, mlp_hidden_size, tp_size)


def _lm_batch_for_step(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
    data_rank: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + data_rank * 100_003 + step * 1_000_003
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )
    return token_ids[:, :-1].to(device), token_ids[:, 1:].to(device)


def _gather_vocab_logits(local_logits: torch.Tensor, groups: ParallelGroups) -> torch.Tensor:
    gathered = [torch.empty_like(local_logits) for _ in range(groups.tensor_size)]
    dist.all_gather(gathered, local_logits, group=groups.tensor)
    return torch.cat(gathered, dim=-1)


def _sync_lm_gradients(model: TPTransformerLM, groups: ParallelGroups) -> None:
    replicated: list[nn.Parameter] = [model.position_embedding, *list(model.final_norm.parameters())]
    for block in model.blocks:
        replicated.extend(_replicated_parameters(block))
    if groups.tensor_size > 1:
        for parameter in replicated:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=groups.tensor)
    _sync_data_gradients(model, groups)


def _dp_inputs_are_distinct(inputs: torch.Tensor, groups: ParallelGroups) -> bool:
    if groups.data_size <= 1:
        return True
    gathered = [torch.empty_like(inputs) for _ in range(groups.data_size)]
    dist.all_gather(gathered, inputs, group=groups.data)
    return all(not torch.equal(gathered[0], other) for other in gathered[1:])


def _local_parameter_gradient_pairs(
    model: TPTransformerLM,
    reference: ReferenceTransformerLM,
    groups: ParallelGroups,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    actual = dict(model.named_parameters())
    expected = dict(reference.named_parameters())
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def add(name: str, reference_name: str, start: int | None = None, end: int | None = None) -> None:
        actual_parameter = actual[name]
        expected_parameter = expected[reference_name]
        assert actual_parameter.grad is not None and expected_parameter.grad is not None
        gradient = expected_parameter.grad
        if start is not None and end is not None:
            gradient = gradient[start:end]
        pairs.append((name, actual_parameter.grad, gradient))

    local_vocab = model.vocab_size // groups.tensor_size
    vocab_start = groups.tensor_rank * local_vocab
    add("token_embedding.weight", "token_embedding.weight", vocab_start, vocab_start + local_vocab)
    add("position_embedding", "position_embedding")
    for layer_index, block in enumerate(model.blocks):
        prefix = f"blocks.{layer_index}"
        ref_prefix = f"blocks.{layer_index}"
        mappings = (
            ("ln1.weight", "ln1.weight", None),
            ("ln1.bias", "ln1.bias", None),
            ("qkv.q.weight", "q_weight", "row"),
            ("qkv.q.bias", "q_bias", "row"),
            ("qkv.k.weight", "k_weight", "row"),
            ("qkv.k.bias", "k_bias", "row"),
            ("qkv.v.weight", "v_weight", "row"),
            ("qkv.v.bias", "v_bias", "row"),
            ("attention_out.weight", "out_weight", "column"),
            ("attention_bias", "out_bias", None),
            ("ln2.weight", "ln2.weight", None),
            ("ln2.bias", "ln2.bias", None),
            ("up.weight", "up_weight", "row"),
            ("up.bias", "up_bias", "row"),
            ("down.weight", "down_weight", "column"),
            ("down_bias", "down_bias", None),
        )
        for local_name, reference_name, split in mappings:
            actual_name = f"{prefix}.{local_name}"
            reference_full_name = f"{ref_prefix}.{reference_name}"
            start = end = None
            if split == "row":
                size = dict(model.named_parameters())[actual_name].shape[0]
                start = groups.tensor_rank * size
                end = start + size
            elif split == "column":
                size = dict(model.named_parameters())[actual_name].shape[1]
                start = groups.tensor_rank * size
                end = start + size
                gradient = expected[reference_full_name].grad
                assert gradient is not None
                pairs.append((actual_name, actual[actual_name].grad, gradient[:, start:end]))
                continue
            add(actual_name, reference_full_name, start, end)
    add("final_norm.weight", "final_norm.weight")
    add("final_norm.bias", "final_norm.bias")
    add("lm_head.weight", "lm_head.weight", vocab_start, vocab_start + local_vocab)
    return pairs


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float()).item()
    denominator = max(torch.linalg.vector_norm(expected.float()).item(), 1e-2)
    return float(numerator / denominator)


def _lm_correctness_probe(
    model: TPTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    groups: ParallelGroups,
    context: DistributedContext,
    autocast_dtype: torch.dtype | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference = ReferenceTransformerLM(
        model.weights,
        model.d_model,
        model.blocks[0].n_heads,
        context.device,
    )
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    local_logits = model(inputs, groups.tensor, autocast_dtype)
    gathered_logits = _gather_vocab_logits(local_logits, groups)
    reference_logits = reference(inputs, autocast_dtype)
    parallel_loss = vocab_parallel_cross_entropy(
        local_logits,
        targets,
        vocab_start=model.lm_head.vocab_start,
        vocab_size=model.vocab_size,
        tensor_group=groups.tensor,
    )
    reference_loss = F.cross_entropy(
        reference_logits.float().reshape(-1, model.vocab_size),
        targets.reshape(-1),
    )
    parallel_loss.backward()
    _sync_lm_gradients(model, groups)
    reference_loss.backward()
    if groups.data_size > 1:
        for parameter in reference.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad, group=groups.data)
            parameter.grad.div_(groups.data_size)
    comparisons: dict[str, float] = {
        "forward_max_error": float((gathered_logits.float() - reference_logits.float()).abs().max().item()),
        "loss_error": float(abs(parallel_loss.item() - reference_loss.item())),
    }
    for name, actual, expected in _local_parameter_gradient_pairs(model, reference, groups):
        comparisons[f"{name}_grad_error"] = float((actual.float() - expected.float()).abs().max().item())
    error_tensor = torch.tensor(list(comparisons.values()), dtype=torch.float64, device=context.device)
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    relative = _relative_error(gathered_logits, reference_logits)
    relative_tensor = torch.tensor([relative], dtype=torch.float64, device=context.device)
    dist.all_reduce(relative_tensor, op=dist.ReduceOp.MAX)
    distinct = _dp_inputs_are_distinct(inputs, groups)
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    max_error = float(error_tensor.max().item())
    relative_error = float(relative_tensor.item())
    tolerance = rtol if autocast_dtype is not None else atol
    tolerance_error = relative_error if autocast_dtype is not None else max_error
    status = "ok" if tolerance_error <= tolerance and distinct else "failed"
    result = {
        "forward_max_error": float(error_tensor[0].item()),
        "loss_error": float(error_tensor[1].item()),
        "max_parameter_gradient_error": max(
            float(error_tensor[index].item()) for index in range(2, len(comparisons))
        ),
        "relative_l2_error": relative_error,
        "relative_l2_floor": 1e-2,
        "data_parallel_batches_distinct": distinct,
        "tolerance": tolerance,
        "tolerance_mode": "relative_l2" if autocast_dtype is not None else "absolute",
        "status": status,
    }
    return result if context.is_main else {"status": status}


def _communication_estimate(model: TPTransformerLM, groups: ParallelGroups, args: Any) -> dict[str, int]:
    element_size = 2 if args.precision == "bf16" else 4
    activation_bytes = args.batch_size * args.seq_length * args.d_model * element_size
    tp_activation = 2 * activation_bytes * max(groups.tensor_size - 1, 0) * args.n_layers
    replicated: list[nn.Parameter] = [model.position_embedding, *list(model.final_norm.parameters())]
    for block in model.blocks:
        replicated.extend(_replicated_parameters(block))
    replicated_bytes = sum(parameter.numel() for parameter in replicated) * element_size
    vocab_ce_bytes = args.batch_size * args.seq_length * element_size * 3
    tp_bytes = int(tp_activation + replicated_bytes * max(groups.tensor_size - 1, 0) + vocab_ce_bytes)
    dp_bytes = int(
        sum(parameter.numel() for parameter in model.parameters())
        * element_size
        * max(groups.data_size - 1, 0)
    )
    return {
        "tp_bytes_estimate": tp_bytes,
        "dp_bytes_estimate": dp_bytes,
        "total_bytes_estimate": tp_bytes + dp_bytes,
    }


def tp_dp_lm_train(args: Any) -> dict[str, Any] | None:
    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        validate_transformer_world_size(context.world_size, args.tp_size)
        groups = _build_parallel_groups(context, args.tp_size)
        mlp_hidden_size = args.mlp_hidden_size or 4 * args.d_model
        validate_lm_shape(
            args.vocab_size,
            args.d_model,
            args.n_heads,
            args.n_layers,
            mlp_hidden_size,
            groups.tensor_size,
        )
        if args.precision == "bf16":
            if context.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise ValueError("TP/DP BF16 需要支持 BF16 的 CUDA 设备")
            autocast_dtype: torch.dtype | None = torch.bfloat16
        else:
            autocast_dtype = None
        model = TPTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mlp_hidden_size=mlp_hidden_size,
            tensor_rank=groups.tensor_rank,
            tensor_size=groups.tensor_size,
            device=context.device,
            seed=args.seed,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        inputs, targets = _lm_batch_for_step(
            args.batch_size,
            args.seq_length,
            args.vocab_size,
            args.seed,
            groups.data_rank,
            0,
            context.device,
        )
        probe = _lm_correctness_probe(
            model,
            inputs,
            targets,
            groups,
            context,
            autocast_dtype,
            args.atol,
            args.rtol,
        )
        if probe["status"] != "ok":
            raise RuntimeError(f"vocab-parallel LM correctness probe failed: {probe}")
        context.barrier()
        measured: list[float] = []
        final_loss = 0.0
        for step in range(args.warmup_steps + args.steps):
            inputs, targets = _lm_batch_for_step(
                args.batch_size,
                args.seq_length,
                args.vocab_size,
                args.seed,
                groups.data_rank,
                step + 1,
                context.device,
            )
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            local_logits = model(inputs, groups.tensor, autocast_dtype)
            loss = vocab_parallel_cross_entropy(
                local_logits,
                targets,
                vocab_start=model.lm_head.vocab_start,
                vocab_size=model.vocab_size,
                tensor_group=groups.tensor,
            )
            loss.backward()
            _sync_lm_gradients(model, groups)
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
        loss_tensor = torch.tensor(final_loss, dtype=torch.float64, device=context.device)
        if groups.tensor_rank == 0:
            dist.reduce(loss_tensor, dst=0, group=groups.data)
            if context.is_main:
                final_loss = float(loss_tensor.item() / groups.data_size)
        if not context.is_main:
            return None
        mean_step_ms = statistics.fmean(measured)
        p50 = float(torch.tensor(measured).quantile(0.5).item())
        p95 = float(torch.tensor(measured).quantile(0.95).item())
        global_tokens = args.batch_size * args.seq_length * groups.data_size
        communication = _communication_estimate(model, groups, args)
        payload = {
            "benchmark": "tp_dp_causal_lm_training",
            "benchmark_schema_version": 1,
            "status": "ok",
            "task": "causal_lm",
            "strategy": "tp_dp_transformer_lm",
            "backend": args.backend or ("nccl" if context.device.type == "cuda" else "gloo"),
            "device": str(context.device),
            "precision": args.precision,
            "world_size": context.world_size,
            "tp_degree": groups.tensor_size,
            "dp_degree": groups.data_size,
            "batch_size_per_rank": args.batch_size,
            "seq_length": args.seq_length,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "mlp_hidden_size": mlp_hidden_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "loss": final_loss,
            "perplexity": math.exp(min(final_loss, 20.0)),
            "step_time_ms": mean_step_ms,
            "step_time_p50_ms": p50,
            "step_time_p95_ms": p95,
            "tokens_per_sec": global_tokens / (mean_step_ms / 1000),
            "communication_bytes_per_step_estimate": communication["total_bytes_estimate"],
            "tp_communication_bytes_estimate": communication["tp_bytes_estimate"],
            "dp_communication_bytes_estimate": communication["dp_bytes_estimate"],
            "communication_estimate_scope": "activation, replicated gradient, DP gradient, and vocab CE collectives",
            "correctness": {
                **probe,
                "dp_parameters_consistent_after_step": parameters_consistent,
            },
            "parallel_groups": {
                "tensor_rank_layout": "tp_rank = global_rank % tp_size; dp_rank = global_rank // tp_size",
                "tensor_size": groups.tensor_size,
                "data_size": groups.data_size,
            },
            "limitations": [
                "MHA only; no GQA/MQA",
                "untied token embedding and LM head",
                "no checkpoint/resume or optimizer state sharding",
                "no pipeline parallel or multi-node TP",
            ],
        }
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    finally:
        context.close()
