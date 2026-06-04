import os, time, math, torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from .config import get_default_config, Config
from .data_loader import data_loading
from train_utils.cross_entropy import CrossEntropyLoss


# Check back the tokenizer implementation --> Is the encoding dataset output numpy array as the output?
def open_memmap_1d(
    file_path: str | os.PathLike | Path,
    np_dtype: str = "uint16"
) -> np.memmap:
    """
        Open a 1D token memmap file. The file is assumed to be a raw binary array.
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(file_path)
    
    if nbytes % itemsize != 0:
        raise ValueError(f"File size not divisible by dtype size: {file_path} ({nbytes}, itemsize={itemsize})")
    length = nbytes // itemsize
    
    return np.memmap(file_path, mode="r", dtype=dtype, shape=(length, ))


def torch_dtype_from_string(
    s: str
) -> torch.dtype:
    s = s.lower()
    if s in ["float32", "fp32"]:
        return torch.float32
    elif s in ["float16", "fp16"]:
        return torch.float16
    elif s in ["bfloat16", "bf16"]:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupport torch dtype: {s}")
    

def set_optimizer_lr(
    optimizer: Optimizer, lr: float
):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: Module, data: np.memmap, cfg: Config
) -> float:
    model.eval()
    loss_func = CrossEntropyLoss()
    losses = []

    for _ in range(cfg.train.eval_batches):
        xb, yb = data_loading(
            data,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.torch_device
        )
        logits = model(xb)
        batch, seq_len, vocab = logits.shape
        loss = loss_func(
            logits.reshape(batch*seq_len, vocab), yb.reshape(batch*seq_len)
        )
        losses.append(float(loss.item()))
    
    model.train()
    return float(np.mean(losses))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    """
        Main training loop.
    """

    # Load config and set seed
    cfg = get_default_config()
    set_seed(cfg.train.seed)

    # Prepare filesystem and load dataset
    os.makedirs(
        os.path.dirname(cfg.train.ckpt_path), exist_ok=True
    )
    train_mm = open_memmap_1d(
        cfg.data.train_data_path, cfg.data.dtype
    )
    val_mm = open_memmap_1d(
        cfg.data.val_data_path, cfg.data.dtype
    )
    
    # Create model and device
    device = torch.device(cfg.data.torch_device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)