import torch.nn as nn
import torch


class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        x: torch.Tensor, # [batch, n_vocab]
        target: torch.Tensor # [batch, ]
    ):
        x_shift = x - torch.amax(x, dim=-1, keepdim=True)
        log_softmax = x_shift - torch.log(
            torch.sum(torch.exp(x_shift), dim=-1, keepdim=True)
        )
        log_prob = log_softmax.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -1 * log_prob
        return loss.mean()