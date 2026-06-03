import torch
import os


def save_checkpoint(
    model, optimizer, iteration: int, out: str | os.PathLike
):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()

    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "iteration": iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]