import torch.nn as nn
import torch
import math
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        weights: torch.Tensor | None = None, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        freeze: bool = False
    ):
        super().__init__()

        if weights is None:
            self.weights = nn.Parameter(
                data=torch.empty(
                    out_features, in_features, device=device, dtype=dtype
                ),
                requires_grad=(not freeze)
            )

            std_value = math.sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(
                self.weights, 
                mean=0.0, 
                std=std_value,
                a=-3*std_value,
                b=3*std_value
            )
        else:
            self.weights = nn.Parameter(
                data=weights.to(device=device, dtype=dtype),
                requires_grad=(not freeze)
            )

    def forward(self, x):
        return einsum(
            x, self.weights,
            "... in_dim, out_dim in_dim -> ... out_dim"
        )
