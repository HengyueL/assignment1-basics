import torch.nn as nn
import torch
from transformer_utils.embedding import Embedding
from transformer_utils.transformer_block import TransformerBlock
from transformer_utils.rope import RoPE
from transformer_utils.rmsnorm import RMSNorm
from transformer_utils.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int,
        d_ff: int,
        num_heads: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,

    ):
        super().__init__()
        d_model = embedding_dim
                
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rope = RoPE(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=self.rope) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x