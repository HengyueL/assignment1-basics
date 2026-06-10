"""
    Given an input (text sequence), decode the next token (text string) until:
    1. an <|endoftext|>
    2. or reaches max_seq_len
"""
from pathlib import Path

import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.bpe import BPETokenizer, DOC_SPECIAL_TOKEN
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
        torch.load(checkpoint_path, map_location=device)["model"]
    )
    model.eval()

    context_window = cfg.model.context_length
    return model, device, context_window


def temperature_scaling(
    logits: torch.Tensor, 
    tau: float = 1.
) -> torch.Tensor:
    softmax = SoftMax()
    logits = logits / tau
    return softmax(logits, dim=-1)


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
    prompt: str,
    max_token_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:

    # 1. Load the trained LM and the tokenizer it was trained with.
    lm_model, device, context_window_size = get_model(model_ckpt_path)

    eos_token = DOC_SPECIAL_TOKEN.decode("utf-8")
    tokenizer = BPETokenizer(
        vocab_size=get_default_config().model.vocab_size,
        special_tokens=[eos_token],
    )
    tokenizer.load(tokenizer_ckpt_path)  # overrides vocab/merges/special tokens
    eos_id = tokenizer.special_tokens.get(eos_token)

    # 2. Tokenize the prompt -> [1, seq_len] tensor on the model's device.
    input_tokens = tokenizer._tokenize_chunk(prompt)
    input_tokens_tensor = torch.as_tensor(
        input_tokens, dtype=torch.long, device=device
    ).reshape(1, -1)

    # 3. Autoregressive decoding loop.
    generated_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_token_length):
            # Keep only the last `context_window_size` tokens the model can attend to.
            model_input = input_tokens_tensor[:, -context_window_size:]

            # Logits for every position; only the last one predicts the next token.
            logits = lm_model(model_input)
            next_logits = logits[0, -1, :]

            # Pick the next token id.
            if temperature <= 0.0:
                next_id = torch.argmax(next_logits)
            else:
                probs = temperature_scaling(next_logits, tau=temperature)
                probs = nucleus_sampling(probs, top_p=top_p)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_id_int = int(next_id)
            if next_id_int == eos_id:
                break

            generated_ids.append(next_id_int)
            input_tokens_tensor = torch.cat(
                [input_tokens_tensor, next_id.reshape(1, 1)], dim=-1
            )

    # 4. Decode the generated ids back into text.
    generated_bytes = b"".join(tokenizer.vocab.get(i, b"") for i in generated_ids)
    return generated_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    cfg = get_default_config()
    output = generate(
        model_ckpt_path=str(cfg.train.ckpt_path),
        tokenizer_ckpt_path=str(Path(__file__).resolve().parent / "MyModel.model"),
        prompt="Once upon a time",
        max_token_length=256,
        temperature=0.8,
        top_p=0.95,
    )
    print(output)