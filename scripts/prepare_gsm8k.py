#!/usr/bin/env python3
"""Download and deterministically prepare the official GSM8K JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def download_and_verify(name: str, destination: Path) -> str:
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == SHA256[name]:
            return digest
    if not destination.exists() or digest != SHA256[name]:
        temporary = destination.with_suffix(destination.suffix + ".download")
        subprocess.run(
            [
                "curl",
                "-4",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "180",
                "--speed-limit",
                "1024",
                "--speed-time",
                "30",
                "-A",
                "MiniTrainBench/0.2",
                "-o",
                str(temporary),
                URLS[name],
            ],
            check=True,
        )
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != SHA256[name]:
            raise ValueError(f"{name} SHA-256 mismatch: {digest} != {SHA256[name]}")
        temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"{name} SHA-256 mismatch: {digest} != {SHA256[name]}")
    return digest


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.train_samples <= 0:
        raise SystemExit("--train-samples must be positive")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw = {}
    digests = {}
    for name in ("train", "test"):
        path = root / f"{name}.source.jsonl"
        digests[name] = download_and_verify(name, path)
        raw[name] = load_jsonl(path)
    if args.train_samples > len(raw["train"]):
        raise SystemExit("--train-samples exceeds the official train split")
    indices = sorted(random.Random(args.seed).sample(range(len(raw["train"])), args.train_samples))
    for name, rows in (("train", [raw["train"][i] for i in indices]), ("test", raw["test"])):
        with (root / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                output = {"id": f"{name}-{indices[index] if name == 'train' else index}", **row}
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "openai/gsm8k",
        "license": "MIT",
        "source_urls": URLS,
        "source_sha256": digests,
        "official_counts": {"train": len(raw["train"]), "test": len(raw["test"])},
        "prepared_counts": {"train": len(indices), "test": len(raw["test"])},
        "train_sample_seed": args.seed,
        "train_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
