from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16_384
    seq_length: int = 512
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    """Pre-norm causal Transformer block，用作 FSDP 自动包装单元。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden_states, src_mask=causal_mask, is_causal=True)


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig, activation_checkpointing: bool = False) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.seq_length, config.seq_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.seq_length:
            raise ValueError(
                f"sequence length {sequence_length} 超过配置上限 "
                f"{self.config.seq_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, mask)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
