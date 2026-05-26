import torch.nn as nn
import torch
from .attention import ScaledDotProductAttention
from .rope import RoPE
from einops import einsum, rearrange


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, num_heads: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype = torch.float32,
        freeze: bool = False
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Q = self._init_parameter(dim=d_model, freeze=freeze)
        self.K = self._init_parameter(dim=d_model, freeze=freeze)
        self.V = self._init_parameter(dim=d_model, freeze=freeze)
        self.W0 = self._init_parameter(dim=d_model, freeze=freeze)
        
        self.attention = ScaledDotProductAttention()

    def _init_parameter(self, dim: int, freeze: bool = False):
        param = nn.Parameter(
            data=torch.empty(dim, dim, device=self.device, dtype=self.dtype),
            requires_grad=(not freeze)
        )
        nn.init.trunc_normal_(param, mean=0., std=0.2)
        return param


    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        k = einsum(
            self.K, x,
            "d_k d_model, batch n_seq d_model -> batch n_seq d_k" 
        )
        q = einsum(
            self.Q, x,
            "d_k d_model, batch n_seq d_model -> batch n_seq d_k" 
        )
        v = einsum(
            self.V, x,
            "d_k d_model, batch n_seq d_model -> batch n_seq d_k" 
        )

        k = rearrange(k, "... n_seq (num_heads d) -> ... num_heads n_seq d", num_heads=self.num_heads)
        q = rearrange(q, "... n_seq (num_heads d) -> ... num_heads n_seq d", num_heads=self.num_heads)
        v = rearrange(v, "... n_seq (num_heads d) -> ... num_heads n_seq d", num_heads=self.num_heads)

        if getattr(self, "rope", None) and positions is not None:
            positions = positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        
        n_seq = x.shape[-2]
        mask = torch.ones(n_seq, n_seq, device=self.device).tril()

        multi_head_attention = rearrange(
            self.attention(k=k, q=q, v=v, mask=mask),
            "... num_heads n_seq d -> ... n_seq (num_heads d)"
        )

        return einsum(
            self.W0, multi_head_attention,
            "d_k d_model, ... n_seq d_model -> ... n_seq d_k"
        )