from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import torch

PROVENANCE_ENV_KEYS = {
    "git_revision": "MINITRAINBENCH_GIT_REVISION",
    "image_ref": "MINITRAINBENCH_IMAGE_REF",
    "image_id": "MINITRAINBENCH_IMAGE_ID",
    "base_image": "MINITRAINBENCH_BASE_IMAGE",
    "build_revision": "MINITRAINBENCH_BUILD_REVISION",
    "command": "MINITRAINBENCH_COMMAND",
}


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _git_revision() -> str | None:
    return os.environ.get("MINITRAINBENCH_GIT_REVISION") or _command_output(
        ["git", "rev-parse", "HEAD"]
    )


def _git_dirty() -> bool | None:
    raw = os.environ.get("MINITRAINBENCH_GIT_DIRTY")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes"}
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _nccl_version() -> str | None:
    if not hasattr(torch.cuda, "nccl"):
        return None
    try:
        raw = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if isinstance(raw, tuple):
        return ".".join(str(part) for part in raw)
    return str(raw)


def _driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    output = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return output.splitlines()[0].strip() if output else None


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu = None
    if cuda_available and local_rank < gpu_count:
        gpu = torch.cuda.get_device_name(local_rank)
    cudnn = torch.backends.cudnn.version() if cuda_available else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(cudnn) if cudnn is not None else None,
        "nccl": _nccl_version(),
        "driver": _driver_version(),
        "gpu": gpu,
        "gpu_count": gpu_count,
    }


def collect_provenance() -> dict[str, Any]:
    values: dict[str, Any] = {
        name: os.environ.get(environment_name)
        for name, environment_name in PROVENANCE_ENV_KEYS.items()
    }
    values["git_revision"] = values["git_revision"] or _git_revision()
    values["git_dirty"] = _git_dirty()
    missing = [
        name
        for name in PROVENANCE_ENV_KEYS
        if values.get(name) in (None, "", "unknown")
    ]
    if values["git_dirty"] is not False:
        missing.append("clean_worktree")
    if (
        values.get("git_revision")
        and values.get("build_revision")
        and values["git_revision"] != values["build_revision"]
    ):
        missing.append("revision_match")
    values["missing_fields"] = sorted(set(missing))
    values["complete"] = not values["missing_fields"]
    return values


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    environment = collect_environment()
    environment.update(payload.get("environment", {}))
    payload["environment"] = environment
    existing = payload.get("provenance")
    if isinstance(existing, dict) and existing.get("complete") is True:
        payload["provenance"] = existing
    else:
        payload["provenance"] = collect_provenance()
    return payload
