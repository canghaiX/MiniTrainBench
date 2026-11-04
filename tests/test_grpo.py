from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minitrainbench.cli import build_parser
from minitrainbench.grpo import (
    _group_advantages,
    _grpo_objective,
    _incremental_group_logprobs,
    _prepare_batch,
    _read_rollouts,
    _selected_logprobs,
    _validate_gsm8k_bindings,
    model_fingerprint,
    train_grpo,
)
from minitrainbench.gsm8k import gsm8k_exact_match


def test_gsm8k_reward_is_bound_to_dataset_and_generated_ids():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def decode(self, tokens, **kwargs):
            return "".join(chr(token) for token in tokens)

    tokenizer = Tokenizer()
    rows = [{"id": "train-1", "question": "2+2?", "answer": "#### 4"}]
    groups = [
        {
            "group_id": "train-1",
            "answer": "#### 4",
            "prompt_text": "2+2?",
            "prompt_tokens": tokenizer.encode("2+2?"),
            "completions": [
                {
                    "response_tokens": tokenizer.encode("#### 4"),
                    "response_text": "#### 4",
                    "reward": 1.0,
                }
            ],
        }
    ]
    _validate_gsm8k_bindings(groups, rows, tokenizer)
    groups[0]["completions"][0]["response_text"] = "#### 5"
    with pytest.raises(ValueError, match="generated token IDs"):
        _validate_gsm8k_bindings(groups, rows, tokenizer)
    groups[0]["answer"] = "#### 5"
    with pytest.raises(ValueError, match="prepared dataset"):
        _validate_gsm8k_bindings(groups, rows, tokenizer)


def _groups():
    return [
        {
            "type": "prompt_group",
            "group_id": str(index),
            "prompt_tokens": prompt,
            "target_tokens": [3, 5, 8],
            "completions": [
                {
                    "response_tokens": [3, 5, 8],
                    "old_logprobs": [-3.0] * 3,
                    "reward": 1.0,
                    "finish_reason": "length",
                },
                {
                    "response_tokens": [7, 2],
                    "old_logprobs": [-3.0] * 2,
                    "reward": 0.0,
                    "finish_reason": "eos",
                },
            ],
        }
        for index, prompt in enumerate(([1, 4], [1, 6, 9, 10]))
    ]


def test_group_advantages_and_zero_variance():
    groups = _groups()
    groups[1]["completions"][0]["reward"] = 0.0
    advantages, count = _group_advantages(groups, torch.device("cpu"))
    assert advantages == [1.0, -1.0, 0.0, 0.0]
    assert count == 1


def test_per_completion_grpo_backward_matches_group_mean_gradient():
    old = [torch.tensor([-1.2, -0.7]), torch.tensor([-0.9]), torch.tensor([-1.5, -0.8])]
    reference = [torch.tensor([-1.1, -0.8]), torch.tensor([-0.85]), torch.tensor([-1.4, -0.9])]
    advantages = [1.0, -1.0, 0.5]
    group_indices = (0, 0, 1)
    full_new = [
        torch.tensor([-1.0, -0.8], requires_grad=True),
        torch.tensor([-0.95], requires_grad=True),
        torch.tensor([-1.3, -0.9], requires_grad=True),
    ]
    full_loss, _ = _grpo_objective(
        full_new,
        reference,
        old,
        advantages,
        group_indices,
        clip_epsilon=0.2,
        kl_coef=0.01,
        group_count=2,
    )
    full_loss.backward()
    full_gradients = [value.grad.clone() for value in full_new]

    split_new = [value.detach().clone().requires_grad_() for value in full_new]
    for index, (new, old_value, reference_value, advantage, group_index) in enumerate(
        zip(split_new, old, reference, advantages, group_indices)
    ):
        group_size = group_indices.count(group_index)
        sequence_loss, _ = _grpo_objective(
            [new],
            [reference_value],
            [old_value],
            [advantage],
            (0,),
            clip_epsilon=0.2,
            kl_coef=0.01,
            group_count=1,
        )
        sequence_loss.div(2 * group_size).backward()

    for actual, expected in zip((value.grad for value in split_new), full_gradients):
        torch.testing.assert_close(actual, expected)


def test_gsm8k_reward_is_numeric_exact_match():
    assert gsm8k_exact_match("2 * 9 = 18\n#### 18", "#### 18") == 1.0
    assert gsm8k_exact_match("#### 19", "#### 18") == 0.0


def test_qwen3_tiny_gsm8k_artifact_trains_one_step(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=63,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).eval()
    model_path = tmp_path / "qwen3"
    model.save_pretrained(model_path)
    groups = []
    for index, prompt in enumerate(([1, 4], [2, 6])):
        group = {
            "type": "prompt_group",
            "group_id": str(index),
            "task": "gsm8k",
            "prompt_tokens": list(prompt),
            "target_tokens": [3],
            "target_text": "#### 4",
            "answer": "#### 4",
            "completions": [
                {
                    "response_tokens": [3, 5],
                    "response_text": "#### 4",
                    "old_logprobs": [],
                    "reward": 1.0,
                    "finish_reason": "length",
                },
                {
                    "response_tokens": [6, 7],
                    "response_text": "#### 3",
                    "old_logprobs": [],
                    "reward": 0.0,
                    "finish_reason": "length",
                },
            ],
        }
        groups.append(group)
    inputs, mask, _, _, _ = _prepare_batch(groups, torch.device("cpu"))
    with torch.no_grad():
        logprobs = iter(
            _selected_logprobs(
                model(input_ids=inputs, attention_mask=mask).logits, inputs, groups, temperature=0.8
            )
        )
    for group in groups:
        for completion in group["completions"]:
            completion["old_logprobs"] = next(logprobs).tolist()
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": hashlib.sha256(b"token_ids:vocab=64").hexdigest(),
        "tokenizer_mode": "token_ids",
        "vocab_size": 64,
        "sampling": {"temperature": 0.8, "top_p": 1.0, "seed": 7},
        "generation": {"num_generations": 2, "max_new_tokens": 2, "eos_token_ids": [63]},
        "reward_function": "gsm8k_exact_match_v1",
        "prompt_group_count": 2,
    }
    rollout = tmp_path / "qwen-rollout.jsonl"
    rollout.write_text("\n".join(json.dumps(row) for row in [manifest, *groups]) + "\n")
    args = build_parser().parse_args(
        [
            "rl",
            "grpo-train",
            "--model-path",
            str(model_path),
            "--rollouts",
            str(rollout),
            "--output-model",
            str(tmp_path / "updated"),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--precision",
            "fp32",
            "--steps",
            "1",
            "--groups-per-step",
            "1",
            "--learning-rate",
            "0.001",
        ]
    )
    args.attention_implementation = "sdpa"
    result = train_grpo(args)
    assert result["optimizer_updated"]
    # Qwen3 uses the same DCP contract, including group accumulation and RNG state.
    args.checkpoint_dir = str(tmp_path / "qwen-checkpoints")
    args.save_every = 1
    args.groups_per_step = 2
    args.microbatch_groups = 1
    args.output_model = str(tmp_path / "first-qwen")
    train_grpo(args)
    args.resume = "latest"
    args.steps = 2
    args.output_model = str(tmp_path / "resumed-qwen")
    resumed = train_grpo(args)
    args.resume = None
    args.checkpoint_dir = str(tmp_path / "continuous-checkpoints")
    args.output_model = str(tmp_path / "continuous-qwen")
    continuous = train_grpo(args)
    assert resumed["policy_state_sha256"] == continuous["policy_state_sha256"]


def test_grpo_clipping_kl_and_sequence_weighting():
    current = [
        torch.log(torch.tensor([1.5], requires_grad=True)),
        torch.log(torch.tensor([0.5, 0.5, 0.5], requires_grad=True)),
    ]
    old = [torch.zeros_like(values) for values in current]
    reference = [values.detach() + 0.1 for values in current]
    loss, metrics = _grpo_objective(
        current,
        reference,
        old,
        [1.0, -1.0],
        (0, 0),
        clip_epsilon=0.2,
        kl_coef=0.1,
        group_count=1,
    )
    expected_kl = torch.exp(torch.tensor(0.1)) - 0.1 - 1
    expected = (-1.2 + 0.8) / 2 + 0.1 * expected_kl
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["kl"], expected_kl)
    assert metrics["clip_fraction"].item() == 1.0
    loss.backward()


def test_response_mask_excludes_prompt_and_padding_but_keeps_eos():
    groups = _groups()
    inputs, mask, _, _, _ = _prepare_batch(groups, torch.device("cpu"))
    logits = torch.randn(*inputs.shape, 16, requires_grad=True)
    actual = _selected_logprobs(logits, inputs, groups, temperature=0.8)
    assert [item.numel() for item in actual] == [3, 2, 3, 2]
    sum(item.sum() for item in actual).backward()
    row = 0
    for group in groups:
        prompt_len = len(group["prompt_tokens"])
        for completion in group["completions"]:
            end = prompt_len - 1 + len(completion["response_tokens"])
            assert torch.count_nonzero(logits.grad[row, : prompt_len - 1]) == 0
            assert torch.count_nonzero(logits.grad[row, end:]) == 0
            assert logits.grad[row, end - 1].abs().sum() > 0
            row += 1
    assert mask[1].sum().item() == 4


@pytest.fixture
def artifact(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(2026)
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval()
    model_path = tmp_path / "policy"
    model.save_pretrained(model_path)
    groups = _groups()
    inputs, mask, _, _, _ = _prepare_batch(groups, torch.device("cpu"))
    with torch.no_grad():
        logits = model(input_ids=inputs, attention_mask=mask).logits
        logprobs = iter(_selected_logprobs(logits, inputs, groups, temperature=0.8))
    for group in groups:
        for completion in group["completions"]:
            completion["old_logprobs"] = next(logprobs).tolist()
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": "LlamaForCausalLM",
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": hashlib.sha256(b"token_ids:vocab=16").hexdigest(),
        "tokenizer_mode": "token_ids",
        "vocab_size": 16,
        "sampling": {"temperature": 0.8, "top_p": 1.0, "seed": 2026},
        "generation": {"num_generations": 2, "max_new_tokens": 3, "eos_token_ids": [2]},
        "reward_function": "token_prefix_accuracy_v1",
        "prompt_group_count": 2,
    }
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [manifest, *groups]) + "\n")
    return model_path, path


def _hf_reference_artifact(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(231)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=63,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3ForCausalLM(config).eval()
    model_path = tmp_path / "hf-reference-qwen3"
    model.save_pretrained(model_path)
    groups = [
        {
            "type": "prompt_group",
            "group_id": str(index),
            "prompt_tokens": list(prompt),
            "target_tokens": [3, 5],
            "completions": [
                {
                    "response_tokens": [3, 5, 9],
                    "old_logprobs": [],
                    "reward": 2.0 / 3.0,
                    "finish_reason": "length",
                },
                {
                    "response_tokens": [6, 7, 8],
                    "old_logprobs": [],
                    "reward": 0.0,
                    "finish_reason": "length",
                },
            ],
        }
        for index, prompt in enumerate(([1, 4], [2, 6, 9]))
    ]
    with torch.no_grad():
        logprobs = iter(
            _incremental_group_logprobs(model, groups, torch.device("cpu"), temperature=0.8)
        )
        for group in groups:
            for completion in group["completions"]:
                completion["old_logprobs"] = next(logprobs).tolist()
    manifest = {
        "type": "manifest",
        "schema_version": 1,
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "engine": "hf_reference",
        "forward_contract": {
            "name": "qwen3_hf_incremental_cache_v1",
            "attention_implementation": "sdpa",
            "weights_dtype": "float32",
            "autocast_dtype": "float32",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "logprob_convention": "temperature_scaled_untruncated_policy",
        "model_fingerprint": model_fingerprint(model_path),
        "tokenizer_fingerprint": hashlib.sha256(b"token_ids:vocab=64").hexdigest(),
        "tokenizer_mode": "token_ids",
        "vocab_size": 64,
        "sampling": {"temperature": 0.8, "top_p": 1.0, "seed": 231},
        "generation": {
            "num_generations": 2,
            "max_new_tokens": 3,
            "eos_token_ids": [63],
        },
        "reward_function": "token_prefix_accuracy_v1",
        "prompt_group_count": len(groups),
    }
    path = tmp_path / "hf-reference-rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [manifest, *groups]) + "\n")
    return model_path, path


def _arguments(artifact, tmp_path, label, *, steps=1, resume=False):
    model_path, rollout_path = artifact
    arguments = [
        "rl",
        "grpo-train",
        "--model-path",
        str(model_path),
        "--rollouts",
        str(rollout_path),
        "--output-model",
        str(tmp_path / label),
        "--output",
        str(tmp_path / f"{label}.json"),
        "--device",
        "cpu",
        "--backend",
        "gloo",
        "--precision",
        "fp32",
        "--steps",
        str(steps),
        "--learning-rate",
        "0.001",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--save-every",
        "1",
    ]
    if resume:
        arguments.extend(["--resume", "latest"])
    manifest = json.loads(rollout_path.read_text().splitlines()[0])
    if manifest.get("engine") == "hf_reference":
        arguments.extend(["--attention-implementation", "sdpa"])
    return arguments


def test_hf_incremental_preflight_gradient_and_exact_resume(tmp_path):
    artifact = _hf_reference_artifact(tmp_path)
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "preflight"))
    args.preflight_only = True
    preflight = train_grpo(args)
    assert preflight["status"] == "ok"
    assert preflight["policy_forward_mode"] == "qwen3_hf_incremental_cache_v1"
    assert preflight["logprob_probe"]["max_abs_error"] <= 1e-6
    assert not preflight["optimizer_updated"]

    args = build_parser().parse_args(_arguments(artifact, tmp_path, "continuous", steps=2))
    args.groups_per_step = 2
    args.microbatch_groups = 1
    args.checkpoint_dir = str(tmp_path / "continuous-checkpoints")
    continuous = train_grpo(args)
    assert continuous["optimizer_updated"]
    assert continuous["dp_parameters_consistent"]

    first_args = build_parser().parse_args(
        _arguments(artifact, tmp_path, "first-incremental", steps=1)
    )
    first_args.groups_per_step = 2
    first_args.microbatch_groups = 1
    first_args.checkpoint_dir = str(tmp_path / "resume-checkpoints")
    first_args.save_every = 1
    train_grpo(first_args)
    resumed_args = build_parser().parse_args(
        _arguments(artifact, tmp_path, "resumed-incremental", steps=2, resume=True)
    )
    resumed_args.groups_per_step = 2
    resumed_args.microbatch_groups = 1
    resumed_args.checkpoint_dir = str(tmp_path / "resume-checkpoints")
    resumed_args.save_every = 1
    resumed = train_grpo(resumed_args)
    assert resumed["resumed"]
    assert resumed["policy_state_sha256"] == continuous["policy_state_sha256"]


def test_matching_preflight_report_is_reused_for_update(tmp_path):
    artifact = _hf_reference_artifact(tmp_path)
    preflight_args = build_parser().parse_args(_arguments(artifact, tmp_path, "gate"))
    preflight_args.preflight_only = True
    train_grpo(preflight_args)

    args = build_parser().parse_args(_arguments(artifact, tmp_path, "reused-gate"))
    args.groups_per_step = 2
    args.microbatch_groups = 1
    args.preflight_report = str(tmp_path / "gate.json")
    result = train_grpo(args)

    assert result["optimizer_updated"]
    assert result["logprob_probe"]["max_abs_error"] <= 1e-6


def test_hf_incremental_rejects_runtime_contract_mismatch(tmp_path):
    artifact = _hf_reference_artifact(tmp_path)
    records = [json.loads(line) for line in artifact[1].read_text().splitlines()]
    records[0]["forward_contract"]["transformers_version"] = "0.0.invalid"
    artifact[1].write_text("\n".join(json.dumps(row) for row in records) + "\n")
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "contract-mismatch"))

    with pytest.raises(ValueError, match="forward contract mismatch"):
        train_grpo(args)
    assert not Path(args.output_model).exists()


def test_hf_incremental_two_process_gloo_updates_consistent_parameters(tmp_path):
    artifact = _hf_reference_artifact(tmp_path)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "minitrainbench",
            *_arguments(artifact, tmp_path, "incremental-ddp"),
        ],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "incremental-ddp.json").read_text())
    assert report["optimizer_updated"]
    assert report["dp_parameters_consistent"]
    assert report["logprob_probe"]["passed"]


def test_rollout_validation_rejects_corrupt_tokens_rewards_and_logprobs(artifact, tmp_path):
    _, path = artifact
    original = path.read_text()
    for field, value, message in (
        ("old_logprobs", [float("nan")] * 3, "invalid completion"),
        ("response_tokens", [99, 5, 8], "invalid response token"),
        ("reward", 0.5, "reward does not match"),
    ):
        records = [json.loads(line) for line in original.splitlines()]
        records[1]["completions"][0][field] = value
        corrupt = tmp_path / "corrupt.jsonl"
        corrupt.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        with pytest.raises(ValueError, match=message):
            _read_rollouts(corrupt)


def test_grpo_cpu_updates_and_exact_checkpoint_resume(artifact, tmp_path):
    continuous = train_grpo(
        build_parser().parse_args(_arguments(artifact, tmp_path / "continuous", "policy", steps=2))
    )
    first = train_grpo(build_parser().parse_args(_arguments(artifact, tmp_path, "first")))
    resumed = train_grpo(
        build_parser().parse_args(_arguments(artifact, tmp_path, "resumed", steps=2, resume=True))
    )
    assert first["optimizer_updated"]
    assert first["initial_logprob_max_error"] < 1e-5
    assert first["dp_parameters_consistent"]
    assert resumed["resumed"]
    assert resumed["global_step"] == 2
    assert resumed["policy_state_sha256"] == continuous["policy_state_sha256"]


def test_grpo_rejects_policy_fingerprint_mismatch(artifact, tmp_path):
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "bad"))
    config_path = artifact[0] / "config.json"
    config_path.write_text(config_path.read_text() + "\n")
    with pytest.raises(ValueError, match="policy fingerprint"):
        train_grpo(args)


def test_failed_preflight_writes_diagnostics_without_updating(artifact, tmp_path):
    records = [json.loads(line) for line in artifact[1].read_text().split("\n") if line]
    records[1]["completions"][0]["old_logprobs"][0] -= 1.0
    artifact[1].write_text("\n".join(json.dumps(row) for row in records) + "\n")
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "failed"))
    with pytest.raises(ValueError, match="logprob mismatch"):
        train_grpo(args)
    report = json.loads(Path(args.output).read_text())
    assert report["status"] == "failed"
    assert report["logprob_probe"]["max_abs_error"] > 0.99
    assert not report["optimizer_updated"]
    assert not Path(args.output_model).exists()


def test_preflight_only_preserves_the_policy(artifact, tmp_path):
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "probe"))
    args.preflight_only = True
    result = train_grpo(args)
    assert result["status"] == "ok"
    assert result["logprob_probe"]["passed"]
    assert not result["optimizer_updated"]
    assert not Path(args.output_model).exists()


def test_group_microbatch_update_matches_full_batch(artifact, tmp_path):
    transformers = pytest.importorskip("transformers")
    args = build_parser().parse_args(_arguments(artifact, tmp_path, "full"))
    args.groups_per_step = 2
    train_grpo(args)
    args.microbatch_groups = 1
    args.output_model = str(tmp_path / "micro")
    args.checkpoint_dir = str(tmp_path / "micro-checkpoints")
    train_grpo(args)
    full = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "full", local_files_only=True
    )
    micro = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "micro", local_files_only=True
    )
    for expected, actual in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_grpo_two_process_gloo_and_resume(artifact, tmp_path):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    for label, steps, resume in (("ddp", 1, False), ("ddp-resumed", 2, True)):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                "minitrainbench",
                *_arguments(artifact, tmp_path, label, steps=steps, resume=resume),
            ],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads((tmp_path / f"{label}.json").read_text())
        assert result["optimizer_updated"]
        assert result["dp_parameters_consistent"]
        assert result["world_size"] == 2
        assert result["response_tokens_seen"] == 10
