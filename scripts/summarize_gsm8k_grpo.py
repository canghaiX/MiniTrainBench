#!/usr/bin/env python3
"""Summarize the fixed-seed GSM8K GRPO cycle without external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "values": []}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def _compact(result: dict) -> dict:
    evaluation_fields = (
        "count",
        "correct",
        "exact_match",
        "throughput_samples_per_sec",
        "max_new_tokens",
        "batch_size",
        "seed",
        "temperature",
        "prompt_format",
        "model_fingerprint",
        "tokenizer_fingerprint",
        "data_sha256",
        "sglang_server",
        "provenance",
    )
    training_fields = (
        "command",
        "global_step",
        "world_size",
        "precision",
        "rollout_engine",
        "policy_forward_mode",
        "attention_implementation",
        "prompt_groups",
        "responses_per_group",
        "logprob_probe",
        "optimizer_updated",
        "dp_parameters_consistent",
        "resumed",
        "tokens_per_sec",
        "response_tokens_seen",
        "step_time_ms_mean",
        "peak_memory_allocated_bytes",
        "grad_norm_max",
        "initial_policy_fingerprint",
        "updated_policy_fingerprint",
        "policy_state_sha256",
        "rollout_sha256",
        "data_fingerprint",
        "source_sha256",
        "environment",
        "provenance",
    )
    compact = {
        key: result[key]
        for key in (
            "schema_version",
            "task",
            "experiment_scope",
            "full_cycle_accepted",
            "data_scope",
            "seeds",
            "metrics",
            "rollouts",
            "sglang_container_image_id",
        )
    }
    compact.update(
        {
            "initial": {
                k: result["initial"][k] for k in evaluation_fields if k in result["initial"]
            },
            "updated": [
                {k: item[k] for k in evaluation_fields if k in item} for item in result["updated"]
            ],
            "training": [
                {k: item[k] for k in training_fields if k in item} for item in result["training"]
            ],
            "limitations": [
                "HF reference rollout is a serial correctness baseline, not a throughput optimization.",
                "Single-seed results do not establish stability across sampling seeds."
                if len(result["seeds"]) == 1
                else "Results apply only to the recorded seeds and configuration.",
                "Subset training and fixed token-budget evaluation are not a standard benchmark claim.",
            ],
        }
    )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    initial_path = args.root / "eval_initial.json"
    eval_paths = sorted(args.root.glob("eval_seed_*.json"))
    train_paths = sorted(args.root.glob("seed_*/train.json"))
    evaluations = [_read(path) for path in eval_paths]
    trainings = [_read(path) for path in train_paths]
    if args.expected_seeds <= 0:
        raise ValueError("expected-seeds must be positive")
    if (
        len(evaluations) != args.expected_seeds
        or len(trainings) != args.expected_seeds
        or not initial_path.is_file()
    ):
        raise ValueError(
            "summary requires initial evaluation and one training/evaluation result per seed"
        )
    metadata = _read(args.root / "data/manifest.json")
    initial = _read(initial_path)
    expected_test_count = metadata["prepared_counts"]["test"]
    if any(item["count"] != expected_test_count for item in [initial, *evaluations]):
        raise ValueError("evaluation count does not match prepared test split")
    if any(
        item.get("status") != "ok"
        or not item.get("optimizer_updated")
        or not item.get("logprob_probe", {}).get("passed")
        for item in trainings
    ):
        raise ValueError("cannot summarize failed/preflight-only runs as completed GRPO updates")
    for item in trainings:
        probe = item["logprob_probe"]
        error = probe.get("max_abs_error")
        tolerance = probe.get("tolerance", 0.5)
        if (
            not isinstance(error, (int, float))
            or not math.isfinite(error)
            or error < 0
            or error > 0.5
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or not 0 <= tolerance <= 0.5
            or error > tolerance
            or item.get("global_step") != 16
            or item.get("rollout_engine") != "hf_reference"
            or item.get("policy_forward_mode") != "qwen3_hf_incremental_cache_v1"
        ):
            raise ValueError(
                "summary requires a passed HF reference 16-step run with max error <= 0.5"
            )
        if item.get("initial_policy_fingerprint") == item.get("updated_policy_fingerprint"):
            raise ValueError("completed GRPO evidence requires a changed policy fingerprint")
    train_seeds = [path.parent.name.removeprefix("seed_") for path in train_paths]
    eval_seeds = [path.stem.removeprefix("eval_seed_") for path in eval_paths]
    if train_seeds != eval_seeds:
        raise ValueError("training and evaluation seeds do not match")
    if any(not item["dp_parameters_consistent"] for item in trainings):
        raise ValueError("DDP parameters did not agree")
    server_fields = (
        "version",
        "dtype",
        "attention_backend",
        "sampling_backend",
        "enable_fp32_lm_head",
        "enable_deterministic_inference",
        "rl_on_policy_target",
    )
    for item in [initial, *evaluations]:
        server = item.get("sglang_server", {})
        if (
            item.get("temperature") != 0.0
            or server.get("version") != "0.5.18"
            or server.get("attention_backend") != "fa3"
        ):
            raise ValueError("summary requires greedy SGLang 0.5.18/FA3 evaluation")
        score = item["exact_match"]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("evaluation exact match must be finite and within [0, 1]")
    if any(
        item["data_sha256"] != initial["data_sha256"]
        or item["prompt_format"] != initial["prompt_format"]
        or item["max_new_tokens"] != initial["max_new_tokens"]
        or item["tokenizer_fingerprint"] != initial["tokenizer_fingerprint"]
        or item.get("batch_size") != initial.get("batch_size")
        or item.get("seed") != initial.get("seed")
        or any(
            item["sglang_server"].get(key) != initial["sglang_server"].get(key)
            for key in server_fields
        )
        for item in evaluations
    ):
        raise ValueError(
            "evaluation dataset/prompt/tokenizer/generation/server settings do not match"
        )
    for training, evaluation in zip(trainings, evaluations):
        if training["updated_policy_fingerprint"] != evaluation["model_fingerprint"]:
            raise ValueError("evaluation model does not match the updated policy")
        if training["initial_policy_fingerprint"] != initial["model_fingerprint"]:
            raise ValueError("training initial policy does not match the baseline model")
    rollout_manifests = []
    for path, training in zip(train_paths, trainings):
        rollout_path = path.parent / "rollout.jsonl"
        if rollout_path.is_file():
            if hashlib.sha256(rollout_path.read_bytes()).hexdigest() != training.get(
                "rollout_sha256"
            ):
                raise ValueError("rollout artifact does not match the training fingerprint")
            with rollout_path.open(encoding="utf-8") as stream:
                manifest = json.loads(next(stream))
            if manifest.get("type") != "manifest":
                raise ValueError("rollout header must be a manifest")
            rollout_manifests.append(manifest)
    image_id_path = args.root / "sglang_image_id.json"
    result = {
        "schema_version": 1,
        "task": "gsm8k_grpo",
        "experiment_scope": "single_seed_exploratory" if args.expected_seeds == 1 else "multi_seed",
        "full_cycle_accepted": True,
        "data_scope": {
            "train": metadata["prepared_counts"]["train"],
            "test": expected_test_count,
            "source_sha256": metadata["source_sha256"],
        },
        "initial": initial,
        "seeds": [path.stem.removeprefix("eval_seed_") for path in eval_paths],
        "updated": evaluations,
        "training": trainings,
        "rollouts": rollout_manifests,
        "sglang_container_image_id": _read(image_id_path) if image_id_path.is_file() else None,
        "metrics": {
            "exact_match": _stats([float(item["exact_match"]) for item in evaluations]),
            "exact_match_delta_percentage_points": _stats(
                [
                    100.0 * (float(item["exact_match"]) - float(initial["exact_match"]))
                    for item in evaluations
                ]
            ),
            "throughput_samples_per_sec": _stats(
                [float(item["throughput_samples_per_sec"]) for item in evaluations]
            ),
            "initial_logprob_max_error": _stats(
                [
                    float(item["initial_logprob_max_error"])
                    for item in trainings
                    if "initial_logprob_max_error" in item
                ]
            ),
            "peak_memory_allocated_bytes": _stats(
                [
                    float(item["peak_memory_allocated_bytes"])
                    for item in trainings
                    if "peak_memory_allocated_bytes" in item
                ]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.compact_output is not None:
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(json.dumps(_compact(result), indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
