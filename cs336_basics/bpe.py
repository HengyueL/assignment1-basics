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
        vocab = self._build_vocab()
        num_merges = self.vocab_size - 256 - len(self.special_tokens)

        aggregated: dict[tuple[int, ...], int] = {}
        for file_name in pretoken_files:
            file_path = pretokenization_path / file_name
            with open(file_path, "rb") as f:
                pretoken_dict = json.load(f)
            for key, value in pretoken_dict.items():
                tok = tuple(key.encode("utf-8"))
                aggregated[tok] = aggregated.get(tok, 0) + value

        # Inverted index pair -> word_ids; each merge visits only the affected words
        # instead of the whole corpus, dropping cost from O(M*N) to ~O(N + M*k).
        word_tokens: dict[int, list[int]] = {}
        word_count: dict[int, int] = {}
        pair_counts: dict[tuple[int, int], int] = {}
        pair_to_words: dict[tuple[int, int], set[int]] = {}

        for word_id, (toks, c) in enumerate(aggregated.items()):
            word_tokens[word_id] = list(toks)
            word_count[word_id] = c
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + c
                pair_to_words.setdefault(p, set()).add(word_id)

        for merge_idx in range(num_merges):
            if not pair_counts:
                self.logger.info(f"No pairs left at merge {merge_idx}; stopping early.")
                break

            best_pair = max(
                pair_counts,
                key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]),
            )
            merge_id = 256 + merge_idx
            merge[best_pair] = merge_id
            vocab[merge_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            a, b = best_pair

            # Snapshot: the loop body mutates pair_to_words[best_pair].
            affected = list(pair_to_words.get(best_pair, ()))

            for wid in affected:
                old_tokens = word_tokens[wid]
                c = word_count[wid]

                for i in range(len(old_tokens) - 1):
                    p = (old_tokens[i], old_tokens[i + 1])
                    new_count = pair_counts.get(p, 0) - c
                    if new_count <= 0:
                        pair_counts.pop(p, None)
                    else:
                        pair_counts[p] = new_count

                new_tokens: list[int] = []
                i, n = 0, len(old_tokens)
                while i < n:
                    if i + 1 < n and old_tokens[i] == a and old_tokens[i + 1] == b:
                        new_tokens.append(merge_id)
                        i += 2
                    else:
                        new_tokens.append(old_tokens[i])
                        i += 1
                word_tokens[wid] = new_tokens

                old_pairs = {(old_tokens[i], old_tokens[i + 1]) for i in range(len(old_tokens) - 1)}
                new_pairs: set[tuple[int, int]] = set()
                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + c
                    new_pairs.add(p)

                for p in new_pairs - old_pairs:
                    pair_to_words.setdefault(p, set()).add(wid)
                for p in old_pairs - new_pairs:
                    idx_set = pair_to_words.get(p)
                    if idx_set is not None:
                        idx_set.discard(wid)
                        if not idx_set:
                            del pair_to_words[p]

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