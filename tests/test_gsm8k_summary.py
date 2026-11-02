import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _run(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/summarize_gsm8k_grpo.py"
    spec = importlib.util.spec_from_file_location("summarize_gsm8k_grpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary", "--root", str(tmp_path), "--output", str(tmp_path / "summary.json")],
    )
    module.main()
    return json.loads((tmp_path / "summary.json").read_text())


def _write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def experiment(tmp_path):
    _write(
        tmp_path,
        "data/manifest.json",
        {
            "prepared_counts": {"train": 256, "test": 1319},
            "source_sha256": {},
        },
    )
    evaluation = {
        "count": 1319,
        "data_sha256": "test-data",
        "prompt_format": "same-prompt",
        "max_new_tokens": 256,
        "tokenizer_fingerprint": "same-tokenizer",
        "model_fingerprint": "initial",
        "exact_match": 0.5,
        "throughput_samples_per_sec": 10.0,
    }
    _write(tmp_path, "eval_initial.json", evaluation)
    for seed, score in ((2026, 0.5), (2027, 0.6), (2028, 0.7)):
        model = f"updated-{seed}"
        _write(
            tmp_path,
            f"eval_seed_{seed}.json",
            {
                **evaluation,
                "model_fingerprint": model,
                "exact_match": score,
            },
        )
        _write(
            tmp_path,
            f"seed_{seed}/train.json",
            {
                "status": "ok",
                "optimizer_updated": True,
                "logprob_probe": {"passed": True},
                "dp_parameters_consistent": True,
                "initial_policy_fingerprint": "initial",
                "updated_policy_fingerprint": model,
                "initial_logprob_max_error": 0.2,
                "peak_memory_allocated_bytes": 100,
            },
        )
    return tmp_path


def test_complete_three_seed_summary(experiment, monkeypatch):
    result = _run(experiment, monkeypatch)
    assert result["metrics"]["exact_match"]["mean"] == pytest.approx(0.6)
    assert result["metrics"]["exact_match"]["std"] == pytest.approx(0.1)
    assert result["data_scope"]["test"] == 1319


def test_failed_preflight_cannot_be_presented_as_training(experiment, monkeypatch):
    _write(
        experiment,
        "seed_2026/train.json",
        {
            "status": "failed",
            "optimizer_updated": False,
            "logprob_probe": {"passed": False},
        },
    )
    with pytest.raises(ValueError, match="failed/preflight-only"):
        _run(experiment, monkeypatch)
    assert not (experiment / "summary.json").exists()


def test_wrong_evaluated_model_is_rejected(experiment, monkeypatch):
    path = experiment / "eval_seed_2026.json"
    data = json.loads(path.read_text())
    data["model_fingerprint"] = "initial"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match the updated policy"):
        _run(experiment, monkeypatch)
