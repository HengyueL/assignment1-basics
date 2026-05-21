import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype | None = torch.float32,
        device: torch.device | None = None,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
        w3: torch.Tensor | None = None,
        freeze: bool = False
    ):
        super().__init__()
        self.freeze = freeze
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = self._init_weight(w1)
        self.w2 = self._init_weight(w2)
        self.w3 = self._init_weight(w3)


    def _init_weight(
        self, w: torch.Tensor | None
    ):
        if w is None:
            weights = nn.Parameter(
                data=torch.empty(self.d_ff, self.d_model, device=self.device, dtype=self.dtype),
                requires_grad=(not self.freeze)
            )
            nn.init.trunc_normal_(weights, mean=0, std=0.2)
        else:
            weights = nn.Parameter(
                data=w.to(device=self.device, dtype=self.dtype),
                requires_grad=(not self.freeze)
            )
        return weights

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = self._silu(
            einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        )
        x_w3 = einsum(
            x, self.w3, "... d_model, d_ff d_model -> ... d_ff"
        )
        right = silu * x_w3

        return einsum(
            self.w2, right, "d_ff d_model, ... d_ff -> ... d_model"
        )

