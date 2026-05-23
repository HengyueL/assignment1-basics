import torch
import torch.nn.functional as F
import torch.nn as nn


class SoftMax(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor, dim: int):
        x = x - torch.amax(x, dim=dim, keepdim=True)
        x_exp = torch.exp(x)
        return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
