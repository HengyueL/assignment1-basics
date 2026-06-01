from torch.optim import Optimizer
from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        assert lr > 0, f"Invalid learning rate: {lr}"
        assert eps >= 0, f"Invalid eps: {eps}"
        assert weight_decay >= 0, f"Invalid weight_decay: {weight_decay}"
        assert 0 <= betas[0] < 1, f"Invalid beta_1: {betas[0]}"
        assert 0 <= betas[1] < 1, f"Invalid beta_2: {betas[1]}"
        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "betas": betas,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta_1, beta_2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialization
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["bc1"] = beta_1  # running beta_1 ** t
                    state["bc2"] = beta_2  # running beta_2 ** t

                m, v = state["m"], state["v"]
                bc1, bc2 = state["bc1"], state["bc2"]

                # Corrected learning rate (bias correction folded into alpha)
                alpha = lr * math.sqrt(1 - bc2) / (1 - bc1)

                # Update biased moment estimates in place
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                # Decoupled weight decay (uses lr, not the corrected alpha)
                p.mul_(1 - lr * weight_decay)
                # Adam step
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha)

                # Advance the running powers of the betas
                state["bc1"] = bc1 * beta_1
                state["bc2"] = bc2 * beta_2

        return loss
