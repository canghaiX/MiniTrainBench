from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch

from .runtime import _write_json

NCCL_ENV_KEYS = (
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
)


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


def _interface_ipv4(name: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "dev", name],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    addresses = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if "inet" in parts:
            addresses.append(parts[parts.index("inet") + 1])
    return addresses


def _network_interfaces() -> list[dict[str, Any]]:
    root = Path("/sys/class/net")
    interfaces = []
    if not root.is_dir():
        return interfaces
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        interfaces.append(
            {
                "name": path.name,
                "operstate": (path / "operstate").read_text().strip()
                if (path / "operstate").is_file()
                else "unknown",
                "mtu": int((path / "mtu").read_text().strip())
                if (path / "mtu").is_file()
                else None,
                "ipv4": _interface_ipv4(path.name),
            }
        )
    return interfaces


def _check_connectivity(
    master_addr: str | None,
    master_port: int | None,
    timeout: float,
    skip: bool,
) -> dict[str, Any]:
    if skip or not master_addr or not master_port:
        return {"status": "skipped", "reason": "未提供 master addr/port 或显式跳过"}
    try:
        with socket.create_connection((master_addr, master_port), timeout=timeout):
            return {
                "status": "ok",
                "master_addr": master_addr,
                "master_port": master_port,
                "timeout_seconds": timeout,
            }
    except OSError as error:
        return {
            "status": "failed",
            "master_addr": master_addr,
            "master_port": master_port,
            "timeout_seconds": timeout,
            "reason": str(error),
        }


def _diagnostics(
    args: Any,
    *,
    device_count: int,
    cuda_available: bool,
    interfaces: list[dict[str, Any]],
    connectivity: dict[str, Any],
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = getattr(args, "expected_world_size", 0)
    expected_gpus = getattr(args, "expected_gpus", 0)
    backend = getattr(args, "backend", None)

    if backend == "nccl" and not cuda_available:
        diagnostics.append(
            {
                "level": "error",
                "check": "cuda_for_nccl",
                "message": "请求 NCCL backend，但当前 torch.cuda 不可用。",
            }
        )
    if device_count and local_rank >= device_count:
        diagnostics.append(
            {
                "level": "error",
                "check": "local_rank",
                "message": f"LOCAL_RANK={local_rank} 超过可见 GPU 数 {device_count}。",
            }
        )
    if expected_world_size and expected_world_size != world_size:
        diagnostics.append(
            {
                "level": "warning",
                "check": "world_size",
                "message": (
                    f"期望 WORLD_SIZE={expected_world_size}，当前环境为 {world_size}。"
                ),
            }
        )
    if expected_gpus and device_count < expected_gpus:
        diagnostics.append(
            {
                "level": "warning",
                "check": "gpu_count",
                "message": f"期望至少 {expected_gpus} 张 GPU，当前可见 {device_count} 张。",
            }
        )
    ifname = os.environ.get("NCCL_SOCKET_IFNAME")
    interface_names = {item["name"] for item in interfaces}
    if ifname:
        requested = {name.strip("^") for name in ifname.split(",") if name.strip()}
        if requested.isdisjoint(interface_names):
            diagnostics.append(
                {
                    "level": "warning",
                    "check": "NCCL_SOCKET_IFNAME",
                    "message": f"NCCL_SOCKET_IFNAME={ifname} 未匹配当前网卡列表。",
                }
            )
    if world_size > 1 and connectivity["status"] == "failed":
        diagnostics.append(
            {
                "level": "warning",
                "check": "rdzv_connectivity",
                "message": "MASTER_ADDR/PORT 连通性检查失败，多机 torchrun 可能 hang。",
            }
        )
    if not diagnostics:
        diagnostics.append(
            {
                "level": "info",
                "check": "summary",
                "message": "未发现明确阻塞项；多机仍需在所有节点执行相同 doctor 检查。",
            }
        )
    return diagnostics


def run_doctor(args: Any) -> dict[str, Any]:
    interfaces = _network_interfaces()
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    master_addr = args.master_addr or os.environ.get("MASTER_ADDR")
    master_port = args.master_port
    if master_port is None and os.environ.get("MASTER_PORT"):
        master_port = int(os.environ["MASTER_PORT"])
    connectivity = _check_connectivity(
        master_addr,
        master_port,
        args.timeout,
        args.skip_connectivity,
    )
    cuda_available = torch.cuda.is_available()
    payload = {
        "benchmark": "doctor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "nccl_version": _nccl_version(),
        "gpu_count": device_count,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
            }
            for index in range(device_count)
        ],
        "distributed_env": {
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "master_addr": master_addr,
            "master_port": master_port,
        },
        "nccl_env": {
            key: os.environ[key]
            for key in NCCL_ENV_KEYS
            if key in os.environ
        },
        "network_interfaces": interfaces,
        "connectivity": connectivity,
    }
    payload["diagnostics"] = _diagnostics(
        args,
        device_count=device_count,
        cuda_available=cuda_available,
        interfaces=interfaces,
        connectivity=connectivity,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload
