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
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freq = 1. / (theta ** (k / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, freq)
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cache_cos", cos, persistent=False)
        self.register_buffer("cache_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos_values = self.cache_cos[token_positions]
        sin_values = self.cache_sin[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2]

        rotated_x1 = cos_values * x1 - sin_values * x2
        rotated_x2 = sin_values * x1 + cos_values * x2
        return rearrange(
            torch.stack([rotated_x1, rotated_x2], dim=-1), 
            '... d_half two -> ... (d_half two)'
        )


if __name__ == "__main__":
    rope = RoPE(
        theta=10,
        d_k=4,
        max_seq_len=10
    )
