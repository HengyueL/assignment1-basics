import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple


def data_loading(
    x: npt.NDArray, 
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    starts = np.random.randint(0, len(x)-context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)

    inputs = torch.as_tensor(x[idx], dtype=torch.long, device=device)
    targets = torch.as_tensor(x[idx+1], dtype=torch.long, device=device)
    return inputs, targets