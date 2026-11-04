#!/usr/bin/env python3
"""Delete only verified experiment intermediates; retain reports and rollouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def cleanup(root: Path, phase: str) -> dict:
    root = root.resolve(strict=True)
    if phase == "smoke":
        report = json.loads((root / "smoke/train-resumed.json").read_text())
        if (
            report.get("status") != "ok"
            or report.get("resumed") is not True
            or report.get("global_step") != 2
            or report.get("optimizer_updated") is not True
            or report.get("dp_parameters_consistent") is not True
            or report.get("initial_policy_fingerprint") == report.get("updated_policy_fingerprint")
        ):
            raise ValueError("smoke cleanup requires a verified update and DCP resume")
        targets = [
            root / "smoke" / name for name in ("updated-step1", "updated-resumed", "checkpoints")
        ]
    elif phase == "complete":
        summary = json.loads((root / "summary.json").read_text())
        if summary.get("full_cycle_accepted") is not True:
            raise ValueError("checkpoint cleanup requires an accepted full cycle")
        seeds = summary.get("seeds", [])
        if not seeds or any(not str(seed).isdigit() for seed in seeds):
            raise ValueError("summary contains invalid seed directories")
        targets = [root / f"seed_{seed}" / "checkpoints" for seed in seeds]
    else:
        raise ValueError("unknown cleanup phase")
    # Validate every path before deleting any. Never follow links outside this run.
    for target in targets:
        if target.resolve() != target or target.is_symlink():
            raise ValueError(f"refusing symlink cleanup target: {target}")
        if target.exists() and any(path.is_symlink() for path in target.rglob("*")):
            raise ValueError(f"refusing linked intermediate tree: {target}")
    audit = {"phase": phase, "removed": [], "bytes_removed": 0}
    for target in targets:
        if target.is_dir():
            size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
            shutil.rmtree(target)
            audit["removed"].append(str(target.relative_to(root)))
            audit["bytes_removed"] += size
    (root / f"cleanup_{phase}.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("smoke", "complete"))
    args = parser.parse_args()
    print(json.dumps(cleanup(args.root, args.phase), indent=2))


if __name__ == "__main__":
    main()
