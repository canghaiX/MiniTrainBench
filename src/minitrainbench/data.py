from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SyntheticTokenIterator:
    """按 global_step 生成可复现的 synthetic token batch。"""

    vocab_size: int
    batch_size: int
    seq_length: int
    seed: int
    rank: int

    def batch_for_step(self, global_step: int, device: torch.device) -> torch.Tensor:
        if global_step < 0:
            raise ValueError("global_step 不能为负数")
        generator = torch.Generator(device=device.type)
        generator.manual_seed(
            self.seed + global_step * 1_000_003 + self.rank * 97_003
        )
        return torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_length),
            generator=generator,
            device=device,
        )

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "rank": self.rank}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["seed"]) != self.seed or int(state["rank"]) != self.rank:
            raise ValueError("synthetic data iterator 的 seed 或 rank 与 checkpoint 不匹配")
