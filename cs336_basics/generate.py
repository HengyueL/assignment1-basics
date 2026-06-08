"""
    Given an input (text sequence), decode the next token (text string) until:
    1. an <|endoftext|>
    2. or reaches max_seq_len
"""
import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.bpe import BPETokenizer
from cs336_basics.train import torch_dtype_from_string
from cs336_basics.training_loop_utils.config import get_default_config
from cs336_basics.transformer_utils.softmax import SoftMax


def get_model(
    checkpoint_path
):
    cfg = get_default_config()
    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model
    device = torch.device(cfg.data.torch_device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        embedding_dim=cfg.model.d_model,
        d_ff=d_ff,
        num_heads=cfg.model.num_heads,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        rope_theta=cfg.model.rope_theta
    ).to(device=device, dtype=model_dtype)
    
    model.load_state_dict(
        torch.load(checkpoint_path)["model"]
    )
    return model, device


def temperature_scaling(
    logits: torch.Tensor, 
    tau: float = 1.
) -> torch.Tensor:
    softmax = SoftMax()
    logits = logits / tau
    return softmax(logits)


def nucleus_sampling(
    probs: torch.Tensor,
    top_p: float
) -> torch.Tensor:
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be within (0, 1]. Input top_p value: f{top_p}")

    sorted_probs, sorted_idx = torch.sort(
        probs, descending=True
    )
    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep cumulative prob is <= top_p
    keep = cum <= top_p
    keep[..., 0] = True

    filtered_sorted_probs = sorted_probs * keep.to(sorted_probs.dtype)
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted_probs)
    return out


def generate(
    model_ckpt_path,
    tokenizer_ckpt_path,
    input_text,
    max_token_length: int = 1024,
) -> str:
    
    # Init LM Model
    lm_model, device = get_model(model_ckpt_path)

    # Init Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_ckpt_path)

    # Tokenize str
    input_tokens = tokenizer._tokenize_chunk(input_text)
    input_tokens_tensor = torch.as_tensor(
        input_tokens
    ).reshape([1, -1]).to(device=device)

    with torch.no_grad():
        logits = lm_model(input_tokens_tensor)

    raise NotImplementedError