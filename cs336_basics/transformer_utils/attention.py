import torch.nn as nn
import torch
from einops import einsum
from softmax import SoftMax


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftMax()
    
    def forward(
        self, 
        k: torch.Tensor, # [..., m, d_k]
        q: torch.Tensor, # [..., n, d_k]
        v: torch.Tensor, # [..., m, d_v]
        mask: torch.Tensor | None = None # [n, m]
    ) -> torch.Tensor:
        d_k = k.shape[-1]
        inner = einsum(q, k, "... n d_k, ... m d_k -> ... n m") / (d_k ** 0.5)
        if mask is not None:
            inner = inner.masked_fill(mask==0, float("-inf"))

        return einsum(
            self.softmax(inner, dim=-1), v,
            "... n m, ... m d_v -> ... n d_v" 
        ) 
    
    