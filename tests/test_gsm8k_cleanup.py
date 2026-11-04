import importlib.util
import json
from pathlib import Path

import pytest


def _cleanup(root, phase):
    path = Path(__file__).parents[1] / "scripts/cleanup_gsm8k_artifacts.py"
    spec = importlib.util.spec_from_file_location("cleanup_gsm8k", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cleanup(root, phase)


def _write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_smoke_cleanup_keeps_evidence_and_other_runs(tmp_path):
    _write(
        tmp_path,
        "smoke/train-resumed.json",
        {
            "status": "ok",
            "resumed": True,
            "global_step": 2,
            "optimizer_updated": True,
            "dp_parameters_consistent": True,
            "initial_policy_fingerprint": "old",
            "updated_policy_fingerprint": "new",
        },
    )
    for name in ("updated-step1", "updated-resumed", "checkpoints"):
        _write(tmp_path, f"smoke/{name}/weights", "large-data")
    _write(tmp_path, "smoke/rollout.jsonl", "tokens")
    _write(tmp_path, "seed_2026/checkpoints/weights", "keep")
    audit = _cleanup(tmp_path, "smoke")
    assert len(audit["removed"]) == 3
    assert audit["bytes_removed"] > 0
    assert (tmp_path / "smoke/train-resumed.json").exists()
    assert (tmp_path / "smoke/rollout.jsonl").exists()
    assert (tmp_path / "seed_2026/checkpoints/weights").exists()
    assert _cleanup(tmp_path, "smoke")["removed"] == []


def test_complete_cleanup_requires_accepted_cycle_and_keeps_export(tmp_path):
    _write(tmp_path, "summary.json", {"full_cycle_accepted": False, "seeds": ["2026"]})
    _write(tmp_path, "seed_2026/checkpoints/weights", "discard")
    _write(tmp_path, "seed_2026/updated_model/weights", "keep")
    with pytest.raises(ValueError, match="accepted full cycle"):
        _cleanup(tmp_path, "complete")
    assert (tmp_path / "seed_2026/checkpoints/weights").exists()
    _write(tmp_path, "summary.json", {"full_cycle_accepted": True, "seeds": ["2026"]})
    _cleanup(tmp_path, "complete")
    assert not (tmp_path / "seed_2026/checkpoints").exists()
    assert (tmp_path / "seed_2026/updated_model/weights").exists()


def test_cleanup_rejects_path_traversal_and_symlinks(tmp_path):
    _write(tmp_path, "summary.json", {"full_cycle_accepted": True, "seeds": ["../other"]})
    with pytest.raises(ValueError, match="invalid seed"):
        _cleanup(tmp_path, "complete")
    _write(tmp_path, "summary.json", {"full_cycle_accepted": True, "seeds": ["2026"]})
    _write(tmp_path, "outside/keep", "keep")
    (tmp_path / "seed_2026").mkdir()
    (tmp_path / "seed_2026/checkpoints").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        _cleanup(tmp_path, "complete")
    assert (tmp_path / "outside/keep").exists()
