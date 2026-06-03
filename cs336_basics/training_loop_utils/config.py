from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

ROOT_DIR = "/Users/pipibo/assignment1-basics/cs336_basics"
CONSTANTS = {
    "context_length": 256
}


@dataclass
class DataConfig:
    train_data_path: str | os.PathLike | Path = Path(ROOT_DIR) / "TinyStoriesV2-GPT4-valid_encoding.lib"
    val_data_path: str | os.PathLike | Path = Path(ROOT_DIR) / "TinyStoriesV2-GPT4-valid_encoding.lib"
    
    # Tokenizer dtype
    dtype: str = "uint16"

    context_length: int = CONSTANTS["context_length"]

    torch_device: str = "cpu"
    
@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = CONSTANTS["context_length"]

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8

    d_ff: Optional[int] = None

    rope_theta: float = 10_000.0

    max_seq_len: int = CONSTANTS["context_length"]

    rmsnorm_eps: float = 1e-6

    torch_dtype: str = "float32"

@dataclass
class OptimizerConfig:
    lr_max: float = 3e-4
    lr_min: float = 3e-5

    warmup_iters: int = 200
    cosine_cycle_iters: int = 10_000

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

@dataclass
class TrainingCofing:
    max_step: int = 10_000
    batch_size: int = 64

    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20

    ckpt_interval: int = 1_000
    ckpt_path: str | os.PathLike | Path = Path(ROOT_DIR) / "checkpoints" / "ckpt.pt"

    resume_from: Optional[str] = None

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingCofing = field(default_factory=TrainingCofing)


def get_default_config() -> Config:
    cfg = Config()
    cfg.model.context_length = cfg.data.context_length
    return cfg