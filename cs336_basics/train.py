import os, time, math, torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
import logging

from training_loop_utils.config import get_default_config, Config
from training_loop_utils.data_loader import data_loading
from train_utils.cross_entropy import CrossEntropyLoss
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_utils.adamw import AdamW
from cs336_basics.train_utils.learning_rate_scheduler import cosine_annealing_scheduler
from cs336_basics.training_loop_utils.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.train_utils.grad_clip import clip_grad
logger = logging.getLogger(__name__)
logging.basicConfig()

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

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        embedding_dim=cfg.model.d_model,
        d_ff=d_ff,
        num_heads=cfg.model.num_heads,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        rope_theta=cfg.model.rope_theta
    ).to(device=device, dtype=model_dtype)

    # Optimizer 
    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr_max,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.beta1, cfg.optim.beta2)
    )

    iter_start = 0
    if cfg.train.resume_from is not None and os.path.exists(cfg.train.resume_from):
        iter_start = load_checkpoint(
            src=cfg.train.resume_from, model=model, optimizer=optimizer
        )
    
    # Training Loop Initialization
    best_val = float("inf")
    last_log_t = time.time()

    # Main training loop
    loss_func = CrossEntropyLoss()
    for it in range(iter_start, cfg.train.max_step):

        # Update learning rate 
        lr = cosine_annealing_scheduler(
            t=it, alpha_max=cfg.optim.lr_max, alpha_min=cfg.optim.lr_min,
            T_w=cfg.optim.warmup_iters,
            T_c=cfg.optim.cosine_cycle_iters
        )
        set_optimizer_lr(optimizer=optimizer, lr=lr)

        # Sample a batch of data
        xb, yb = data_loading(
            train_mm, batch_size=cfg.train.batch_size,
            context_length=cfg.model.context_length,
            device=cfg.data.torch_device
        )

        # Forward pass
        logits = model(xb)
        batch, size, vocab = logits.shape
        loss = loss_func(
            logits.reshape(batch*size, vocab),
            yb.reshape(batch*size)
        )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Grad Clipping
        if cfg.optim.grad_clip > 0:
            clip_grad(model.parameters(), cfg.optim.grad_clip, eps=1e-6)
        
        # Update parameters
        optimizer.step()

        # Periodic training metrics logging
        if (it + 1) % cfg.train.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-9)
            tok_s = (cfg.train.batch_size * cfg.data.context_length * cfg.train.log_interval) / dt
            msg = f"It={it+1} loss={loss.item():.4f} lr={lr:.3e} tok/s={tok_s:.1f}"
            logger.info(msg)
            last_log_t = now
        
        # Periodic eval
        if (it + 1) % cfg.train.eval_interval == 0:
            val_loss = estimate_loss(model, val_mm, cfg)
            val_ppl = float(math.exp(val_loss))
            msg = f"[Eval] Iter={it + 1} Val_loss={val_loss:.4f} Val_ppl={val_ppl:.2f}"
            logger.info(msg)

            # Save the best-performing checkpoint
            if val_loss < best_val:
                best_val = val_loss
                best_path = cfg.train.ckpt_path.replace(".pt", ".best.pt")
                save_checkpoint(model, optimizer, it+1, best_path)
        
        # Periodic Checkpointing
        if (it + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(model, optimizer, it+1, cfg.train.ckpt_path)

    # Save final checkpoint
    save_checkpoint(model, optimizer, cfg.train.max_step, cfg.train.ckpt_path)


if __name__ == "__main__":
    main()