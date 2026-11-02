#!/usr/bin/env python3
"""Summarize the fixed-seed GSM8K GRPO cycle without external dependencies."""

from __future__ import annotations

import argparse
import json
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
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    train_seeds = [path.parent.name.removeprefix("seed_") for path in train_paths]
    eval_seeds = [path.stem.removeprefix("eval_seed_") for path in eval_paths]
    if train_seeds != eval_seeds:
        raise ValueError("training and evaluation seeds do not match")
    if any(not item["dp_parameters_consistent"] for item in trainings):
        raise ValueError("DDP parameters did not agree")
    if any(
        item["data_sha256"] != initial["data_sha256"]
        or item["prompt_format"] != initial["prompt_format"]
        or item["max_new_tokens"] != initial["max_new_tokens"]
        or item["tokenizer_fingerprint"] != initial["tokenizer_fingerprint"]
        for item in evaluations
    ):
        raise ValueError("evaluation dataset/prompt/tokenizer/generation settings do not match")
    for training, evaluation in zip(trainings, evaluations):
        if training["updated_policy_fingerprint"] != evaluation["model_fingerprint"]:
            raise ValueError("evaluation model does not match the updated policy")
        if training["initial_policy_fingerprint"] != initial["model_fingerprint"]:
            raise ValueError("training initial policy does not match the baseline model")
    result = {
        "schema_version": 1,
        "task": "gsm8k_grpo",
        "data_scope": {
            "train": metadata["prepared_counts"]["train"],
            "test": expected_test_count,
            "source_sha256": metadata["source_sha256"],
        },
        "initial": initial,
        "seeds": [path.stem.removeprefix("eval_seed_") for path in eval_paths],
        "updated": evaluations,
        "training": trainings,
        "metrics": {
            "exact_match": _stats([float(item["exact_match"]) for item in evaluations]),
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
