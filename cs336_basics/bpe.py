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
from bpe_utils.utils import get_stats
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
            vocab[idx] = special_string
        
        return vocab

    def _register_special_tokens(self, special_tokens: list[str]):
        for idx, special_token in enumerate(special_tokens):
            self.special_tokens[special_token] = self.vocab_size - idx - 1
            self.inverse_special_tokens[self.vocab_size - idx - 1] = special_token

    def train(self, pretokenization_path: str):
        pretokenization_path = Path(pretokenization_path)
        pretoken_files = [f for f in os.listdir(pretokenization_path) if f.endswith(".json")]
        self.logger.info(f"Available pretokenized files: {pretoken_files}")

        vocab = {idx: bytes([idx]) for idx in range(256)}
        merge = {}
        num_merges = self.vocab_size - 256

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
                total_doc[tuple(text_bytes_list)] = value

        vocab_cache = {}
        for merge_idx in range(num_merges):
            # Count in cache about (p0, p1) stats
            for key, value in total_doc.items():
                vocab_cache = get_stats(key, value, vocab_cache)
                
            # Get the most frequet pair
            pair = max(vocab_cache, key=vocab_cache.get)
            merge_id = 256 + merge_idx
            merge[pair] = merge_id
            vocab[merge_id] = vocab[pair[0]] + vocab[pair[1]]
            
            print(pair, merge_id)
            input()

            # Merge
            for key, value in total_doc.items():
                # Merge the key if
                new_key = merge_key(key, pair, new_idx)
                ...



def train(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    pretokenization_path = Path(input_path).resolve().parent / "processed_chunks"
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
    print(bpe_tokenizer.vocab)
    bpe_tokenizer.train(pretokenization_path=pretokenization_path)

if __name__ == "__main__":
    
    input_doc_path = CURRENT_DIR.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1024
    special_tokens = [DOC_SPECIAL_TOKEN.decode("utf-8")]
    
    train(input_doc_path, vocab_size, special_tokens)
    pass