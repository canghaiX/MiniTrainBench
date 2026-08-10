from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool
    rendezvous_file: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
        if self.rendezvous_file:
            try:
                os.unlink(self.rendezvous_file)
            except FileNotFoundError:
                pass


def setup_distributed(
    backend: str | None = None,
    device_name: str = "auto",
    requires_process_group: bool = False,
) -> DistributedContext:
    requested_cuda = device_name == "cuda" or (
        device_name == "auto" and torch.cuda.is_available()
    )
    if requested_cuda and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA，但 torch.cuda.is_available() 返回 false")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank) if requested_cuda else torch.device("cpu")
    chosen_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
    initialized_here = False
    rendezvous_file = None

    if (world_size > 1 or requires_process_group) and not dist.is_initialized():
        if chosen_backend == "nccl":
            torch.cuda.set_device(device)
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            rendezvous_file = os.path.join(
                tempfile.gettempdir(), f"minitrainbench-{os.getpid()}.rendezvous"
            )
            Path(rendezvous_file).unlink(missing_ok=True)
            init_method = f"file://{rendezvous_file}"
        else:
            init_method = "env://"
        dist.init_process_group(
            backend=chosen_backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    elif device.type == "cuda":
        torch.cuda.set_device(device)

    return DistributedContext(
        rank,
        local_rank,
        world_size,
        device,
        initialized_here,
        rendezvous_file,
    )
