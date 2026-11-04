import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _run(tmp_path, monkeypatch, expected_seeds=3, compact_output=False):
    path = Path(__file__).parents[1] / "scripts/summarize_gsm8k_grpo.py"
    spec = importlib.util.spec_from_file_location("summarize_gsm8k_grpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "summary.json"),
            "--expected-seeds",
            str(expected_seeds),
            *(["--compact-output", str(tmp_path / "compact.json")] if compact_output else []),
        ],
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
        "temperature": 0.0,
        "batch_size": 128,
        "seed": 2026,
        "sglang_server": {
            "version": "0.5.18",
            "dtype": "bfloat16",
            "attention_backend": "fa3",
        },
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
                "logprob_probe": {"passed": True, "max_abs_error": 0.2, "tolerance": 0.5},
                "dp_parameters_consistent": True,
                "initial_policy_fingerprint": "initial",
                "updated_policy_fingerprint": model,
                "initial_logprob_max_error": 0.2,
                "peak_memory_allocated_bytes": 100,
                "global_step": 16,
                "rollout_engine": "hf_reference",
                "policy_forward_mode": "qwen3_hf_incremental_cache_v1",
            },
        )
    return tmp_path


def test_complete_three_seed_summary(experiment, monkeypatch):
    result = _run(experiment, monkeypatch)
    assert result["metrics"]["exact_match"]["mean"] == pytest.approx(0.6)
    assert result["metrics"]["exact_match"]["std"] == pytest.approx(0.1)
    assert result["metrics"]["exact_match_delta_percentage_points"]["mean"] == pytest.approx(10)
    assert result["data_scope"]["test"] == 1319


def test_compact_summary_preserves_acceptance_and_fingerprints(experiment, monkeypatch):
    result = _run(experiment, monkeypatch, compact_output=True)
    compact = json.loads((experiment / "compact.json").read_text())
    assert compact["full_cycle_accepted"]
    assert compact["metrics"] == result["metrics"]
    assert compact["training"][0]["logprob_probe"]["max_abs_error"] == 0.2
    assert compact["training"][0]["updated_policy_fingerprint"] == "updated-2026"
    assert "limitations" in compact


def test_compact_summary_keeps_rollout_and_container_provenance(experiment, monkeypatch):
    manifest = {
        "type": "manifest",
        "engine": "hf_reference",
        "provenance": {"source_sha256": "rollout-source"},
    }
    for seed in (2026, 2027, 2028):
        path = experiment / f"seed_{seed}/rollout.jsonl"
        path.write_text(json.dumps(manifest) + "\n")
        train_path = path.parent / "train.json"
        data = json.loads(train_path.read_text())
        data["rollout_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        train_path.write_text(json.dumps(data))
    _write(experiment, "sglang_image_id.json", "sha256:container")
    _run(experiment, monkeypatch, compact_output=True)
    compact = json.loads((experiment / "compact.json").read_text())
    assert compact["sglang_container_image_id"] == "sha256:container"
    assert compact["rollouts"][0]["provenance"]["source_sha256"] == "rollout-source"


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


def test_single_seed_summary_does_not_claim_zero_variance(experiment, monkeypatch):
    for seed in (2027, 2028):
        (experiment / f"eval_seed_{seed}.json").unlink()
        (experiment / f"seed_{seed}" / "train.json").unlink()
    result = _run(experiment, monkeypatch, expected_seeds=1)
    assert result["experiment_scope"] == "single_seed_exploratory"
    assert result["metrics"]["exact_match"]["std"] is None
    assert result["metrics"]["exact_match_delta_percentage_points"]["std"] is None


def test_single_seed_em_regression_is_reported_honestly(experiment, monkeypatch):
    for seed in (2027, 2028):
        (experiment / f"eval_seed_{seed}.json").unlink()
        (experiment / f"seed_{seed}" / "train.json").unlink()
    path = experiment / "eval_seed_2026.json"
    data = json.loads(path.read_text())
    data["exact_match"] = 0.4
    path.write_text(json.dumps(data))
    result = _run(experiment, monkeypatch, expected_seeds=1)
    assert result["full_cycle_accepted"]
    metric = result["metrics"]["exact_match_delta_percentage_points"]
    assert metric["mean"] == pytest.approx(-10)
    assert metric["std"] is None


@pytest.mark.parametrize("error", [0.500001, float("nan"), float("inf"), -0.1])
def test_passed_flag_cannot_override_invalid_gate(experiment, monkeypatch, error):
    path = experiment / "seed_2026/train.json"
    data = json.loads(path.read_text())
    data["logprob_probe"]["max_abs_error"] = error
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="max error <= 0.5"):
        _run(experiment, monkeypatch)
    assert not (experiment / "summary.json").exists()


def test_unchanged_policy_cannot_claim_completed_update(experiment, monkeypatch):
    path = experiment / "seed_2026/train.json"
    data = json.loads(path.read_text())
    data["updated_policy_fingerprint"] = data["initial_policy_fingerprint"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed policy fingerprint"):
        _run(experiment, monkeypatch)
    assert not (experiment / "summary.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_updated", False),
        ("logprob_probe", {"passed": False, "max_abs_error": 0.6}),
    ],
)
def test_incomplete_single_seed_cycle_is_rejected(experiment, monkeypatch, field, value):
    for seed in (2027, 2028):
        (experiment / f"eval_seed_{seed}.json").unlink()
        (experiment / f"seed_{seed}" / "train.json").unlink()
    path = experiment / "seed_2026/train.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _run(experiment, monkeypatch, expected_seeds=1)
    assert not (experiment / "summary.json").exists()


def test_wrong_evaluated_model_is_rejected(experiment, monkeypatch):
    path = experiment / "eval_seed_2026.json"
    data = json.loads(path.read_text())
    data["model_fingerprint"] = "initial"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match the updated policy"):
        _run(experiment, monkeypatch)


@pytest.mark.parametrize("field,value", [("temperature", 0.8), ("batch_size", 1), ("seed", 2027)])
def test_incomparable_decoding_settings_are_rejected(experiment, monkeypatch, field, value):
    path = experiment / "eval_seed_2026.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evaluation|settings"):
        _run(experiment, monkeypatch)
    assert not (experiment / "summary.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("version", "0.5.17"), ("attention_backend", "triton"), ("dtype", "float16")],
)
def test_incomparable_server_settings_are_rejected(experiment, monkeypatch, field, value):
    path = experiment / "eval_seed_2026.json"
    data = json.loads(path.read_text())
    data["sglang_server"][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evaluation|settings"):
        _run(experiment, monkeypatch)
    assert not (experiment / "summary.json").exists()


def test_passed_flag_cannot_override_declared_tolerance(experiment, monkeypatch):
    path = experiment / "seed_2026/train.json"
    data = json.loads(path.read_text())
    data["logprob_probe"]["tolerance"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="max error <= 0.5"):
        _run(experiment, monkeypatch)
