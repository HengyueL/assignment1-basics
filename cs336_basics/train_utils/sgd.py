from torch.optim import Optimizer
from collections.abc import Callable, Iterable
import torch
import math
from typing import Optional


"""
    Notes: any customized variable can be saved to state.
"""

class SGD(Optimizer):
    """
        A learning rate decayed SGD optimizer
    """
    def __int__(self, params, lr=1e-3):
        assert lr > 0, f"Invalid learning rate: {lr}"
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p
                t = state.get("t", 0) # Get the iteration number from the state, default 0
                grad = p.grad.data # Get the gradient w.r.t p

                p.data -= lr / math.sqrt(t+1) * grad # Step decayed learning rate
                state["t"] = t + 1 

        return loss
    

