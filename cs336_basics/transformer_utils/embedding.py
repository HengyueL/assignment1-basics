import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        weights: torch.Tensor | None = None, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        freeze: bool = False
    ):  
        super().__init__()
        
        if weights is None:
            self.weights = nn.Parameter(
                data=torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                requires_grad=(not freeze)
            )
            nn.init.trunc_normal_(
                self.weights, mean=0., std=1, a=-3, b=3
            )
        else:
            self.weights = nn.Parameter(
                data=weights.to(device=device, dtype=dtype),
                requires_grad=(not freeze)
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]