import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    spec = importlib.util.spec_from_file_location(
        "prepare_gsm8k", Path(__file__).parents[1] / "scripts/prepare_gsm8k.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_is_deterministic_and_preserves_full_test(tmp_path, monkeypatch):
    module = _module()
    for name, count in (("train", 10), ("test", 4)):
        contents = "".join(
            json.dumps(
                {"question": f"question\u2028{i}", "answer": f"#### {i}"}, ensure_ascii=False
            )
            + "\n"
            for i in range(count)
        )
        (tmp_path / f"{name}.source.jsonl").write_text(contents, encoding="utf-8")
        monkeypatch.setitem(module.SHA256, name, hashlib.sha256(contents.encode()).hexdigest())
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--output-dir", str(tmp_path), "--train-samples", "3", "--seed", "7"],
    )
    module.main()
    first = (tmp_path / "train.jsonl").read_bytes()
    module.main()
    assert (tmp_path / "train.jsonl").read_bytes() == first
    assert len(module.load_jsonl(tmp_path / "train.jsonl")) == 3
    assert len(module.load_jsonl(tmp_path / "test.jsonl")) == 4
    assert json.loads((tmp_path / "manifest.json").read_text())["prepared_counts"] == {
        "train": 3,
        "test": 4,
    }


def test_download_rejects_wrong_sha256(tmp_path, monkeypatch):
    module = _module()

    def download(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"corrupt")

    monkeypatch.setattr(module.subprocess, "run", download)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.download_and_verify("train", tmp_path / "train.jsonl")
