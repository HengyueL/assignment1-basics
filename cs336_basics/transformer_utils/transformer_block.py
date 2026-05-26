import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .causal_multihead_attention import MultiheadAttention
from .rope import RoPE
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        rope: RoPE | None = None
    ):
        super().__init__()
        
        self.attention = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device
        )
        if rope is not None:
            self.attention.rope = rope

        self.rmsnorm1 = RMSNorm(d_model=d_model)
        self.rmsnorm2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = x + self.attention(
            x=self.rmsnorm1(x),
            positions=positions
        )

        z = y + self.ffn(self.rmsnorm2(y))

        return z