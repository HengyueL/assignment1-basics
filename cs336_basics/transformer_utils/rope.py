import torch.nn as nn
import torch
from einops import rearrange


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    rope = RoPE(
        theta=10,
        d_k=4,
        max_seq_len=10
    )
    print()
