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
    _prepare_batch,
    _read_rollouts,
    _selected_logprobs,
    model_fingerprint,
    train_grpo,
)


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
    return arguments


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
