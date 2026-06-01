from torch.nn import Parameter
from typing import List
import torch


def clip_grad(
    params: List[Parameter],
    l2_max: float,
    eps: float = 1e-6
):  
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    
    grad_norm = torch.sqrt(
        sum((g.detach() ** 2).sum() for g in grads)
    )

    if grad_norm > l2_max:
        scale = l2_max / (grad_norm + eps)
        for g in grads:
            g.mul_(scale)
        