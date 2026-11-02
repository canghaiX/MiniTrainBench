from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import CheckpointManager
from .distributed import DistributedContext, setup_distributed
from .provenance import collect_provenance
from .runtime import TrainState, _write_json
from .scheduler import LearningRateScheduler
from .strategy import DDPStrategy

_MODEL_FILE_SUFFIXES = (".safetensors", ".bin", ".index.json")
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
}


@dataclass(frozen=True)
class GRPOCheckpointConfig:
    strategy: str
    precision: str
    model_config: dict[str, Any]
    training: dict[str, Any]

    def model_dict(self) -> dict[str, Any]:
        return self.model_config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def uses_legacy_runtime_defaults(self) -> bool:
        return False


def _fingerprint_files(root: Path, predicate) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and predicate(path))
    if not files:
        raise ValueError(f"no fingerprintable files found in {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path)
    return _fingerprint_files(
        root,
        lambda path: path.name == "config.json" or path.name.endswith(_MODEL_FILE_SUFFIXES),
    )


def tokenizer_fingerprint(tokenizer_path: str | Path, vocab_size: int) -> str:
    root = Path(tokenizer_path)
    has_tokenizer = any(
        path.is_file() and (path.name in _TOKENIZER_FILES or path.name.endswith(".model"))
        for path in root.rglob("*")
    )
    if not has_tokenizer:
        return hashlib.sha256(f"token_ids:vocab={vocab_size}".encode()).hexdigest()
    return _fingerprint_files(
        root,
        lambda path: path.name in _TOKENIZER_FILES or path.name.endswith(".model"),
    )


def _token_ids_fingerprint(vocab_size: int) -> str:
    return hashlib.sha256(f"token_ids:vocab={vocab_size}".encode()).hexdigest()


def _read_rollouts(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"rollout line {line_number} must be an object")
            records.append(record)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid rollout JSON on line {line_number}: {error}") from error
    if len(records) < 2 or records[0].get("type") != "manifest":
        raise ValueError("rollout file must start with a manifest and contain prompt groups")
    manifest = records[0]
    groups = records[1:]
    if manifest.get("schema_version") != 1 or any(
        item.get("type") != "prompt_group" for item in groups
    ):
        raise ValueError("unsupported or malformed rollout schema")
    if manifest.get("reward_function") != "token_prefix_accuracy_v1":
        raise ValueError("unsupported rollout reward function")
    if manifest.get("architecture") != "LlamaForCausalLM":
        raise ValueError("only Llama rollouts are supported")
    if manifest.get("logprob_convention") != "temperature_scaled_untruncated_policy":
        raise ValueError("unsupported rollout logprob convention")
    sampling = manifest.get("sampling", {})
    if (
        not isinstance(sampling, dict)
        or not isinstance(sampling.get("temperature"), (float, int))
        or not math.isfinite(sampling["temperature"])
        or sampling["temperature"] <= 0
        or not isinstance(sampling.get("top_p"), (float, int))
        or not 0 < sampling["top_p"] <= 1
    ):
        raise ValueError("invalid rollout sampling configuration")
    if manifest.get("tokenizer_mode") not in {"text", "token_ids"}:
        raise ValueError("invalid tokenizer mode")
    if manifest.get("prompt_group_count") != len(groups):
        raise ValueError("manifest prompt_group_count does not match rollout data")
    expected_generations = int(manifest.get("generation", {}).get("num_generations", 0))
    if expected_generations < 2:
        raise ValueError("GRPO needs at least two completions per prompt group")
    seen_ids: set[str] = set()
    vocab_size = manifest.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("invalid rollout vocabulary")
    generation = manifest.get("generation", {})
    max_new_tokens = generation.get("max_new_tokens", 0)
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("invalid rollout generation limit")
    eos_ids = set(generation.get("eos_token_ids", []))
    for group_index, group in enumerate(groups):
        group_id = str(group.get("group_id", group_index))
        if group_id in seen_ids:
            raise ValueError(f"duplicate rollout group_id: {group_id}")
        seen_ids.add(group_id)
        prompt = group.get("prompt_tokens")
        completions = group.get("completions")
        if not isinstance(prompt, list) or not prompt or not isinstance(completions, list):
            raise ValueError(f"malformed prompt group {group_id}")
        if len(completions) != expected_generations:
            raise ValueError(f"prompt group {group_id} has an unexpected completion count")
        target = group.get("target_tokens")
        if not isinstance(target, list) or not target:
            raise ValueError(f"missing reward target in group {group_id}")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in prompt + target):
            raise ValueError(f"invalid token id in group {group_id}")
        for completion in completions:
            if not isinstance(completion, dict):
                raise TypeError(f"invalid completion in prompt group {group_id}")
            response = completion.get("response_tokens")
            old_logprobs = completion.get("old_logprobs")
            reward = completion.get("reward")
            if (
                not isinstance(response, list)
                or not response
                or not isinstance(old_logprobs, list)
                or len(response) != len(old_logprobs)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value > 1e-5
                    for value in old_logprobs
                )
            ):
                raise ValueError(f"invalid completion data in prompt group {group_id}")
            if any(type(token) is not int or not 0 <= token < vocab_size for token in response):
                raise ValueError(f"invalid response token id in group {group_id}")
            reason = completion.get("finish_reason")
            if len(response) > max_new_tokens or reason not in {"eos", "length"}:
                raise ValueError(f"invalid finish metadata in group {group_id}")
            if reason == "eos" and (
                response[-1] not in eos_ids or any(token in eos_ids for token in response[:-1])
            ):
                raise ValueError(f"invalid EOS completion in group {group_id}")
            if reason == "length" and (
                len(response) != max_new_tokens or any(token in eos_ids for token in response)
            ):
                raise ValueError(f"invalid length completion in group {group_id}")
            prefix = 0
            for actual, expected in zip(response, target):
                if actual != expected:
                    break
                prefix += 1
            if abs(reward - prefix / max(len(response), len(target))) > 1e-7:
                raise ValueError(
                    f"reward does not match token_prefix_accuracy_v1 in group {group_id}"
                )
    return manifest, groups, digest


def _load_transformers():
    try:
        from transformers import AutoConfig, LlamaForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "GRPO Llama training requires the optional dependency: pip install 'minitrainbench[rl]'"
        ) from error
    return AutoConfig, LlamaForCausalLM


def _precision_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support BF16")
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {name}")


def _prepare_batch(
    groups: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], list[torch.Tensor], list[float]]:
    sequences = []
    group_indices = []
    old_logprobs = []
    rewards = []
    for group_index, group in enumerate(groups):
        prompt = [int(token) for token in group["prompt_tokens"]]
        for completion in group["completions"]:
            response = [int(token) for token in completion["response_tokens"]]
            sequences.append(prompt + response)
            group_indices.append(group_index)
            old_logprobs.append(torch.tensor(completion["old_logprobs"], dtype=torch.float32))
            rewards.append(float(completion["reward"]))
    width = max(map(len, sequences))
    input_ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask, tuple(group_indices), old_logprobs, rewards


def _selected_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    groups: list[dict[str, Any]],
    *,
    temperature: float,
) -> list[torch.Tensor]:
    if temperature <= 0:
        raise ValueError("rollout sampling metadata is invalid")
    all_new = []
    completion_index = 0
    for group in groups:
        prompt_length = len(group["prompt_tokens"])
        for completion in group["completions"]:
            response = completion["response_tokens"]
            start = prompt_length - 1
            end = start + len(response)
            labels = input_ids[completion_index, start + 1 : end + 1]
            response_logits = logits[completion_index, start:end].float() / temperature
            row_new = (
                torch.log_softmax(response_logits, dim=-1)
                .gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            if labels.numel() != len(response):
                raise ValueError("response token positions exceed the input sequence")
            all_new.append(row_new)
            completion_index += 1
    return all_new


def _group_advantages(
    groups: list[dict[str, Any]], device: torch.device
) -> tuple[list[float], int]:
    advantages = []
    zero_variance_groups = 0
    for group in groups:
        rewards = torch.tensor(
            [float(item["reward"]) for item in group["completions"]],
            dtype=torch.float32,
            device=device,
        )
        std = rewards.std(unbiased=False)
        if not bool(torch.isfinite(std)) or float(std.item()) <= 1e-8:
            normalized = torch.zeros_like(rewards)
            zero_variance_groups += 1
        else:
            normalized = (rewards - rewards.mean()) / std
        advantages.extend(float(value) for value in normalized.cpu().tolist())
    return advantages, zero_variance_groups


def _grpo_objective(
    new_logprobs: list[torch.Tensor],
    reference_logprobs: list[torch.Tensor],
    old_logprobs: list[torch.Tensor],
    advantages: list[float],
    group_indices: tuple[int, ...],
    *,
    clip_epsilon: float,
    kl_coef: float,
    group_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sequence_losses: list[list[torch.Tensor]] = [[] for _ in range(group_count)]
    sequence_kls: list[list[torch.Tensor]] = [[] for _ in range(group_count)]
    clip_values = []
    if not (
        len(new_logprobs)
        == len(reference_logprobs)
        == len(old_logprobs)
        == len(advantages)
        == len(group_indices)
    ):
        raise ValueError("GRPO sequence metadata lengths differ")
    for new, reference, old, advantage, group_index in zip(
        new_logprobs, reference_logprobs, old_logprobs, advantages, group_indices
    ):
        old = old.detach().to(device=new.device, dtype=torch.float32)
        if old.numel() != new.numel():
            raise ValueError("old and current policy logprob lengths differ")
        log_ratio = new.float() - old
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        adv = torch.full_like(ratio, float(advantage))
        policy = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
        ref_delta = reference.detach().float() - new.float()
        sampled_kl = (ref_delta.exp() - ref_delta - 1.0).mean()
        sequence_losses[group_index].append(policy + kl_coef * sampled_kl)
        sequence_kls[group_index].append(sampled_kl.detach())
        clip_values.append((ratio.detach() - 1.0).abs() > clip_epsilon)
    group_losses = [torch.stack(losses).mean() for losses in sequence_losses]
    group_kls = [torch.stack(values).mean() for values in sequence_kls]
    loss = torch.stack(group_losses).mean()
    metrics = {
        "kl": torch.stack(group_kls).mean(),
        "clip_fraction": torch.cat(clip_values).float().mean(),
    }
    return loss, metrics


def _reduce_scalar(value: torch.Tensor, context: DistributedContext, op=dist.ReduceOp.SUM) -> float:
    result = value.detach().to(device=context.device, dtype=torch.float64).clone()
    if context.world_size > 1:
        dist.all_reduce(result, op=op)
        if op == dist.ReduceOp.SUM:
            result /= context.world_size
    return float(result.item())


def _load_rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state().cpu()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verify_rollout_logprobs(
    reference: torch.nn.Module,
    groups: list[dict[str, Any]],
    context: DistributedContext,
    dtype: torch.dtype,
    temperature: float,
    tolerance: float,
) -> float:
    max_error = torch.zeros((), device=context.device)
    with torch.no_grad():
        for group in groups:
            input_ids, mask, _, old_logprobs, _ = _prepare_batch([group], context.device)
            with torch.autocast(
                device_type=context.device.type, dtype=dtype, enabled=dtype != torch.float32
            ):
                logits = reference(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
            actual = _selected_logprobs(logits, input_ids, [group], temperature=temperature)
            error = torch.stack(
                [
                    (new - old.to(context.device)).abs().max()
                    for new, old in zip(actual, old_logprobs)
                ]
            ).max()
            max_error = torch.maximum(max_error, error)
    error_value = _reduce_scalar(max_error, context, op=dist.ReduceOp.MAX)
    if not math.isfinite(error_value) or error_value > tolerance:
        raise ValueError(
            f"rollout/current-policy logprob mismatch: {error_value:.6g} > {tolerance:.6g}"
        )
    return error_value


def train_grpo(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.steps <= 0 or args.groups_per_step <= 0:
        raise ValueError("steps and groups-per-step must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning-rate and max-grad-norm must be positive")
    if not 0 < args.clip_epsilon < 1 or args.kl_coef < 0:
        raise ValueError("clip-epsilon must be in (0,1) and kl-coef cannot be negative")
    if not all(
        math.isfinite(value)
        for value in (
            args.learning_rate,
            args.max_grad_norm,
            args.clip_epsilon,
            args.kl_coef,
            args.logprob_tolerance,
        )
    ):
        raise ValueError("training hyperparameters must be finite")
    manifest, groups, rollout_digest = _read_rollouts(args.rollouts)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    expected_model = model_fingerprint(args.model_path)
    if manifest.get("model_fingerprint") != expected_model:
        raise ValueError("rollout policy fingerprint does not match --model-path")
    if Path(args.output_model).resolve() == Path(args.model_path).resolve():
        raise ValueError("output-model must differ from the input policy directory")
    vocab_size = int(manifest["vocab_size"])
    expected_tokenizer = (
        tokenizer_fingerprint(args.tokenizer_path or args.model_path, vocab_size)
        if manifest.get("tokenizer_mode") == "text"
        else _token_ids_fingerprint(vocab_size)
    )
    if manifest.get("tokenizer_fingerprint") != expected_tokenizer:
        raise ValueError("rollout tokenizer fingerprint does not match the local tokenizer")
    if int(manifest.get("generation", {}).get("num_generations", 0)) < 2:
        raise ValueError("GRPO requires at least two responses per prompt")

    context = setup_distributed(args.backend, args.device, requires_process_group=True)
    try:
        if len(groups) % context.world_size:
            raise ValueError(
                "prompt group count must be divisible by world size so DDP ranks have equal batches"
            )
        local_groups = groups[context.rank :: context.world_size]
        if len(local_groups) < args.groups_per_step:
            raise ValueError("each rank needs at least groups-per-step prompt groups")
        for group in groups:
            for token in group["prompt_tokens"]:
                if not 0 <= int(token) < vocab_size:
                    raise ValueError("rollout prompt contains out-of-vocabulary token")
            for completion in group["completions"]:
                if any(not 0 <= int(token) < vocab_size for token in completion["response_tokens"]):
                    raise ValueError("rollout response contains out-of-vocabulary token")

        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        dtype = _precision_dtype(args.precision, context.device)
        config_class, model_class = _load_transformers()
        model_config = config_class.from_pretrained(args.model_path, local_files_only=True)
        if model_config.model_type != "llama":
            raise ValueError("GRPO currently supports local Llama checkpoints only")
        model_config._attn_implementation = "eager"
        actor = model_class.from_pretrained(
            args.model_path,
            config=model_config,
            local_files_only=True,
            dtype=torch.float32,
        ).to(context.device)
        reference_config = config_class.from_pretrained(args.model_path, local_files_only=True)
        reference_config._attn_implementation = "eager"
        reference = model_class.from_pretrained(
            args.model_path,
            config=reference_config,
            local_files_only=True,
            dtype=torch.float32,
        ).to(context.device)
        max_length = int(actor.config.max_position_embeddings)
        if any(
            len(group["prompt_tokens"])
            + max(map(len, [item["response_tokens"] for item in group["completions"]]))
            > max_length
            for group in groups
        ):
            raise ValueError("rollout sequence exceeds model position limit")
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        actor.eval()
        reference.eval()
        strategy = DDPStrategy()
        actor = strategy.wrap_model(actor, context, args.precision)
        actor_model = actor.module if isinstance(actor, DistributedDataParallel) else actor
        optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=0.0)
        scheduler = LearningRateScheduler(
            optimizer,
            schedule="constant",
            base_lr=args.learning_rate,
        )
        config = GRPOCheckpointConfig(
            strategy="ddp",
            precision=args.precision,
            model_config=actor_model.config.to_dict(),
            training={
                "model_fingerprint": expected_model,
                "tokenizer_fingerprint": expected_tokenizer,
                "rollout_sha256": rollout_digest,
                "learning_rate": args.learning_rate,
                "clip_epsilon": args.clip_epsilon,
                "kl_coef": args.kl_coef,
                "groups_per_step": args.groups_per_step,
                "sampling": manifest["sampling"],
                "seed": args.seed,
                "max_grad_norm": args.max_grad_norm,
            },
        )
        sampling = manifest["sampling"]
        temperature = float(sampling["temperature"])
        initial_logprob_error = _verify_rollout_logprobs(
            reference,
            local_groups,
            context,
            dtype,
            temperature,
            args.logprob_tolerance or (5e-2 if dtype == torch.bfloat16 else 2e-3),
        )
        checkpoint_manager = CheckpointManager(args.checkpoint_dir, context)
        state = TrainState(seed=args.seed, config_fingerprint=config.fingerprint())
        resumed = False
        if args.resume:
            if not checkpoint_manager.enabled:
                raise ValueError("--resume requires --checkpoint-dir")
            resume_target = args.resume
            if resume_target == "latest":
                latest = checkpoint_manager.find_latest()
                if latest is None:
                    raise ValueError("no READY GRPO checkpoint found")
                resume_target = str(latest)
            loaded = checkpoint_manager.load(
                resume_target, actor, optimizer, scheduler, config, state
            )
            state = loaded.train_state
            if scheduler.completed_steps != state.global_step:
                raise ValueError("checkpoint scheduler/global step mismatch")
            if loaded.rng_state is not None:
                torch.set_rng_state(loaded.rng_state["cpu"].cpu())
                if context.device.type == "cuda":
                    torch.cuda.set_rng_state(loaded.rng_state["cuda"].cpu(), context.device)
            resumed = True
        if args.steps <= state.global_step:
            raise ValueError("steps must exceed the checkpoint global step")

        initial_digest = _model_digest(actor_model)
        loss_history = []
        kl_history = []
        clip_history = []
        reward_history = []
        zero_variance_total = 0
        response_tokens_seen = 0
        step_times = []
        grad_norm_history = []
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        for step in range(state.global_step, args.steps):
            _sync_device(context.device)
            step_started = time.perf_counter()
            start_index = (step * args.groups_per_step) % len(local_groups)
            selected_groups = [
                local_groups[(start_index + offset) % len(local_groups)]
                for offset in range(args.groups_per_step)
            ]
            input_ids, attention_mask, group_indices, old_logprobs, rewards = _prepare_batch(
                selected_groups, context.device
            )
            advantages, zero_variance = _group_advantages(selected_groups, context.device)
            zero_variance_total += int(
                _reduce_scalar(torch.tensor(zero_variance, device=context.device), context)
                * context.world_size
            )
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = dtype != torch.float32
            with torch.autocast(
                device_type=context.device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                actor_outputs = actor(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                new_logprobs = _selected_logprobs(
                    actor_outputs.logits,
                    input_ids,
                    selected_groups,
                    temperature=temperature,
                )
                with torch.no_grad():
                    reference_outputs = reference(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    reference_logprobs = _selected_logprobs(
                        reference_outputs.logits,
                        input_ids,
                        selected_groups,
                        temperature=temperature,
                    )
                loss, metrics = _grpo_objective(
                    new_logprobs,
                    reference_logprobs,
                    old_logprobs,
                    advantages,
                    group_indices,
                    clip_epsilon=args.clip_epsilon,
                    kl_coef=args.kl_coef,
                    group_count=len(selected_groups),
                )
            finite_loss = _reduce_scalar(torch.isfinite(loss).int(), context, op=dist.ReduceOp.MIN)
            if not finite_loss:
                raise FloatingPointError(f"non-finite GRPO loss at step {step}")
            loss.backward()
            grad_norm = strategy.clip_grad_norm(actor, args.max_grad_norm)
            finite = torch.tensor(
                int(torch.isfinite(grad_norm).item()), dtype=torch.int32, device=context.device
            )
            if context.world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite gradient detected across DDP ranks")
            optimizer.step()
            scheduler.step()
            token_count = int(
                _reduce_scalar(
                    torch.tensor(sum(len(item) for item in old_logprobs), device=context.device),
                    context,
                )
                * context.world_size
            )
            response_tokens_seen += token_count
            state.global_step += 1
            state.tokens_seen += token_count
            loss_history.append(_reduce_scalar(loss.detach(), context))
            kl_history.append(_reduce_scalar(metrics["kl"], context))
            clip_history.append(_reduce_scalar(metrics["clip_fraction"], context))
            reward_history.append(
                _reduce_scalar(
                    torch.tensor(statistics.fmean(rewards), device=context.device), context
                )
            )
            grad_norm_history.append(_reduce_scalar(grad_norm, context, op=dist.ReduceOp.MAX))
            _sync_device(context.device)
            step_times.append(
                _reduce_scalar(
                    torch.tensor(time.perf_counter() - step_started, device=context.device),
                    context,
                    op=dist.ReduceOp.MAX,
                )
            )
            if (
                checkpoint_manager.enabled
                and args.save_every
                and state.global_step % args.save_every == 0
            ):
                checkpoint_manager.save(
                    actor,
                    optimizer,
                    scheduler,
                    state,
                    config,
                    _load_rng_state(context.device),
                    keep_last=args.keep_last,
                )

        final_digest = _model_digest(actor_model)
        updated = initial_digest != final_digest
        updated_flag = torch.tensor(int(updated), dtype=torch.int32, device=context.device)
        if context.world_size > 1:
            dist.all_reduce(updated_flag, op=dist.ReduceOp.MIN)
        rank_digests = [None] * context.world_size
        if context.world_size > 1:
            dist.all_gather_object(rank_digests, final_digest)
        else:
            rank_digests[0] = final_digest
        if len(set(rank_digests)) != 1:
            raise RuntimeError("GRPO policy parameters differ between DDP ranks")
        context.barrier()
        output_model_path = Path(args.output_model)
        if context.is_main:
            output_model_path.mkdir(parents=True, exist_ok=True)
            actor_model.save_pretrained(output_model_path)
            if manifest["tokenizer_mode"] == "text":
                import shutil

                tokenizer_root = Path(args.tokenizer_path or args.model_path)
                for path in tokenizer_root.iterdir():
                    if path.is_file() and (
                        path.name in _TOKENIZER_FILES or path.name.endswith(".model")
                    ):
                        shutil.copy2(path, output_model_path / path.name)
        context.barrier()
        result = {
            "schema_version": 1,
            "benchmark": "grpo_training",
            "source_sha256": _fingerprint_files(
                Path(__file__).resolve().parents[1], lambda path: path.suffix == ".py"
            ),
            "command": sys.argv,
            "status": "ok",
            "world_size": context.world_size,
            "strategy": "ddp",
            "precision": args.precision,
            "global_step": state.global_step,
            "prompt_groups": len(groups),
            "responses_per_group": int(manifest["generation"]["num_generations"]),
            "initial_policy_fingerprint": expected_model,
            "updated_policy_fingerprint": model_fingerprint(output_model_path),
            "tokenizer_fingerprint": expected_tokenizer,
            "rollout_sha256": rollout_digest,
            "initial_logprob_max_error": initial_logprob_error,
            "loss_mean": statistics.fmean(loss_history),
            "sampled_kl_mean": statistics.fmean(kl_history),
            "clip_fraction_mean": statistics.fmean(clip_history),
            "reward_mean": statistics.fmean(reward_history),
            "zero_variance_groups": zero_variance_total,
            "response_tokens_seen": response_tokens_seen,
            "tokens_seen": state.tokens_seen,
            "step_time_ms_mean": statistics.fmean(step_times) * 1000,
            "tokens_per_sec": (response_tokens_seen / max(sum(step_times), 1e-9)),
            "grad_norm_max": max(grad_norm_history),
            "dp_parameters_consistent": len(set(rank_digests)) == 1,
            "policy_state_sha256": final_digest,
            "peak_memory_allocated_bytes": int(
                _reduce_scalar(
                    torch.tensor(
                        torch.cuda.max_memory_allocated(context.device)
                        if context.device.type == "cuda"
                        else 0,
                        device=context.device,
                    ),
                    context,
                    op=dist.ReduceOp.MAX,
                )
            ),
            "optimizer_updated": bool(updated_flag.item()),
            "checkpoint_dir": args.checkpoint_dir,
            "resumed": resumed,
            "output_model": str(output_model_path),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(context.device),
                "gpu": torch.cuda.get_device_name(context.local_rank)
                if context.device.type == "cuda"
                else None,
            },
            "provenance": collect_provenance(),
        }
        if context.is_main:
            _write_json(args.output, result)
        return result if context.is_main else None
    finally:
        context.close()


def add_grpo_parser(subparsers) -> None:
    parser = subparsers.add_parser("grpo-train", help="训练本地 Llama GRPO rollout")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--groups-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logprob-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--resume", default=None)
