"""
    Implement the bpe tokenizer.
"""
import sys, os
import json
from typing import List
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import logging
logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool
from bpe_utils.pretokenize import (
    pre_tokenize_document,
    RAM_MB,
    N_POOL
)
from bpe_utils.utils import (
    get_stats,
    merge_key
)
DOC_SPECIAL_TOKEN = b"<|endoftext|>"


class BPETokenizer:
    """
        BPE tokenizer class
    """
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        min_vocab_size = 256 + len(special_tokens)
        assert vocab_size > min_vocab_size, f"Vocab size must > {min_vocab_size}"
        
        self.vocab_size = vocab_size
        
        
        # str -> int, e.g. {'<|endoftext|>': 100257}, stored from the end of total vocab
        self.special_tokens = {}
        self.inverse_special_tokens = {}  # Inverse mapping of special tokens
        self._register_special_tokens(special_tokens=special_tokens)  # Use sting

        self.merge = {} # Merge rules: (int, int) -> int
        self.vocab = {} 
        # self._build_vocab()  # Normal vocab use bytes

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RAM: {RAM_MB} mb - N_THREAD: {N_POOL}")
    
    def _build_vocab(self):
        """
            Build the vocab with merge and special characters
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Merge rules
        for (p0, p1), idx in self.merge.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        # Append special characters
        for special_string, idx in self.special_tokens.items():
            vocab[idx] = special_string.encode("utf-8")
        
        return vocab

    def _register_special_tokens(self, special_tokens: list[str]):
        for idx, special_token in enumerate(special_tokens):
            self.special_tokens[special_token] = self.vocab_size - idx - 1
            self.inverse_special_tokens[self.vocab_size - idx - 1] = special_token

    def train(self, pretokenization_path: str, file_prefix: str):
        pretokenization_path = Path(pretokenization_path)
        pretoken_files = [f for f in os.listdir(pretokenization_path) if f.endswith(".json") and f.startswith(file_prefix)]
        self.logger.info(f"Available pretokenized files: {pretoken_files}")

        merge = {}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)

        total_doc = {}
        # Build document chunks and counts
        for file_idx, file_name in enumerate(pretoken_files):
            file_path = pretokenization_path / f"{file_name}"
            with open(file_path, "rb") as file:
                pretoken_dict = json.load(file)
            
            # Main Merge Function
            for key, value in pretoken_dict.items():
                # Convert key to tuple of vocab index
                text_bytes_list = list(key.encode("utf-8"))
                total_doc[tuple(text_bytes_list)] = total_doc.get(tuple(text_bytes_list), 0) + value

        
        for merge_idx in range(num_merges):
            vocab_cache = {}

            # Count in cache about (p0, p1) stats
            for key, value in total_doc.items():
                vocab_cache = get_stats(key, value, vocab_cache)
                
            # Get the most frequet pair
            pair = max(vocab_cache, key=lambda p: (vocab_cache[p], p[0], p[1]))
            merge_id = 256 + merge_idx
            merge[pair] = merge_id

            # Merge
            doc_cache = {}
            for key, value in total_doc.items():
                # Merge the key if
                new_key = merge_key(key, pair, merge_id)
                doc_cache[new_key] = doc_cache.get(new_key, 0) + value
            total_doc = doc_cache
        
        self.merge = merge
        self.vocab = self._build_vocab()


def train(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    input_path = Path(input_path)
    file_prefix = input_path.resolve().stem
    pretokenization_path = input_path.resolve().parent / "processed_chunks"

    pre_tokenize_document(
        input_path=input_path,
        output_path=str(pretokenization_path),
        document_split_bytes=DOC_SPECIAL_TOKEN,
        special_tokens_list=special_tokens
    )
    bpe_tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    bpe_tokenizer.train(
        pretokenization_path=pretokenization_path,
        file_prefix=file_prefix
    )

    vocab = bpe_tokenizer.vocab
    merge = [
        (vocab[p0], vocab[p1]) for (p0, p1) in bpe_tokenizer.merge.keys()
    ]
    return vocab, merge


if __name__ == "__main__":
    
    input_doc_path = CURRENT_DIR.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1024
    special_tokens = [DOC_SPECIAL_TOKEN.decode("utf-8")]
    
    train(input_doc_path, vocab_size, special_tokens)
    pass