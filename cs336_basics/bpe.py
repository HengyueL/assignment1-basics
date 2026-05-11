"""
    Implement the bpe tokenizer.
"""
import sys, os
from pathlib import Path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


class BPETokenizer:
    """
        BPE tokenizer class
    """
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        # {idx: __repr__()}
        self.vocab = self._build_vocab()

        # Merge rules: (int, int) -> int
        self.merge = {}

        # str -> int, e.g. {'<|endoftext|>': 100257}, stored from the end of total vocab
        self.special_tokens = {}
    
    def _build_vocab(self):
        """
            Build the vocab with merge and special characters
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Merge rules
        for (p0, p1), idx in self.merge.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        # Append special characters
        for special_string, idx in self.special_tokens.items():
            self.vocab[idx] = special_string
        
        return vocab

    def train(self):
        raise NotImplementedError