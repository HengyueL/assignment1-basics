import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int,
        eps: float = 1e-5,
        weights: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        freeze: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        if weights is None:
            self.gain = nn.Parameter(
                data=torch.ones(self.d_model, device=device, dtype=dtype),
                requires_grad=(not freeze)
            )
        else:
            self.gain = nn.Parameter(
                data=weights.to(device=device, dtype=dtype),
                requires_grad=(not freeze)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_orig = x.dtype

        x = x.to(dtype=torch.float32)

        rms = torch.sqrt((x**2).mean(dim=-1)+ self.eps).unsqueeze(-1)

        return (x * self.gain / rms).to(dtype=dtype_orig)
