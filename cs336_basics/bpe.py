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
import regex as re
import heapq
import numpy as np
logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool
from bpe_utils.pretokenize import (
    pre_tokenize_document,
    RAM_MB,
    N_POOL,
    GPT2_SPLIT_PATTERN
)
from bpe_utils.utils import (
    render_token
)
DOC_SPECIAL_TOKEN = b"<|endoftext|>"
_ENCODE_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB read buffer for streaming encode


class _RevBytes:
    """Wrap bytes with reversed __lt__ so heapq's min-heap breaks ties by lex-greater bytes."""
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b

    def __lt__(self, other: "_RevBytes") -> bool:
        return self.b > other.b

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RevBytes) and self.b == other.b

    def __hash__(self) -> int:
        return hash(self.b)


class BPETokenizer:
    """
        BPE tokenizer class
    """
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        min_vocab_size = 256 + len(special_tokens)
        assert vocab_size > min_vocab_size, f"Vocab size must > {min_vocab_size}"
        
        self.vocab_size = vocab_size
        self.pattern = GPT2_SPLIT_PATTERN
        
        # str -> int, e.g. {'<|endoftext|>': 100257}, stored from the end of total vocab
        self.special_tokens = {}
        self.inverse_special_tokens = {}  # Inverse mapping of special tokens
        self._register_special_tokens(special_tokens=special_tokens)  # Use sting

        self.merge = {} # Merge rules: (int, int) -> int
        self.vocab = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RAM: {RAM_MB} mb - N_THREAD: {N_POOL}")

        # Used only in inference mode
        self._merge_cache: dict[tuple[int, ...], list[int]] = {}
    
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
    
    def _pretokenize_train(self, input_path: str, pretokenization_path: Path):
        pre_tokenize_document(
            input_path=input_path,
            output_path=str(pretokenization_path),
            document_split_bytes=DOC_SPECIAL_TOKEN,
            special_tokens_list=list(self.special_tokens.keys()),
            split_pattern=self.pattern
        )

    def train(self, input_path: str, pretokenization_path: str, file_prefix: str):
        pretokenization_path = Path(pretokenization_path)
        
        # Pretokenization
        self._pretokenize_train(
            input_path, pretokenization_path
        )

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

        # Lazy-deletion max-heap: push on each count change, discard stale entries
        # (count != pair_counts[p]) when they bubble to the top.
        heap: list = [
            (-count, _RevBytes(vocab[p[0]]), _RevBytes(vocab[p[1]]), p)
            for p, count in pair_counts.items()
        ]
        heapq.heapify(heap)

        for merge_idx in range(num_merges):
            best_pair = None
            while heap:
                neg_count, _, _, p = heap[0]
                if pair_counts.get(p, 0) == -neg_count:
                    heapq.heappop(heap)
                    best_pair = p
                    break
                heapq.heappop(heap)
            if best_pair is None:
                self.logger.info(f"No pairs left at merge {merge_idx}; stopping early.")
                break

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
                        heapq.heappush(
                            heap,
                            (-new_count, _RevBytes(vocab[p[0]]), _RevBytes(vocab[p[1]]), p),
                        )

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
                    new_count = pair_counts.get(p, 0) + c
                    pair_counts[p] = new_count
                    heapq.heappush(
                        heap,
                        (-new_count, _RevBytes(vocab[p[0]]), _RevBytes(vocab[p[1]]), p),
                    )
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

    def _encode_merge(self, seq: List[int]) -> List[int]:
        """
            Merge a sequence until no merges can happen.
        """
        key = tuple(seq)
        if key in self._merge_cache:
            return self._merge_cache[key]
        
        while len(seq) >= 2:
            best_pair, best_id = None, float("inf")
            for i in range(len(seq) - 1):
                # Iteratively find the pair that has the greatest stats in merge rule
                pair = (seq[i], seq[i+1])
                mid = self.merge.get(pair)
                if mid is not None and mid < best_id:
                    best_id = mid
                    best_pair = pair
                
            if best_pair is None:
                # No pair to merge anymore
                break
            
            # Apply merge
            new_seq: List[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    new_seq.append(best_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        
        self._merge_cache[key] = seq
        return seq
            
    def _last_safe_cut(self, buffer: str) -> int:
        """Return the last position in buffer after which it is safe to cut.

        Safe means: the cut falls after a complete special token, or after a
        complete regex pre-token whose match ends strictly before len(buffer)
        (so we never cut in the middle of a word that continues in the next chunk).
        Returns 0 if no safe position is found yet.
        """
        safe = 0
        if self.special_tokens:
            sp_pat = "|".join(re.escape(s) for s in self.special_tokens)
            prev = 0
            for m in re.finditer(sp_pat, buffer):
                # Regex matches in the non-special segment before this token
                seg_offset = prev
                for pm in re.finditer(self.pattern, buffer[prev:m.start()]):
                    end = seg_offset + pm.end()
                    if end < len(buffer):
                        safe = max(safe, end)
                # Special token end is always a safe cut
                safe = max(safe, m.end())
                prev = m.end()
            # Trailing non-special segment
            for pm in re.finditer(self.pattern, buffer[prev:]):
                end = prev + pm.end()
                if end < len(buffer):
                    safe = max(safe, end)
        else:
            for m in re.finditer(self.pattern, buffer):
                if m.end() < len(buffer):
                    safe = max(safe, m.end())
        return safe

    def _tokenize_chunk(self, chunk: str) -> list[int]:
        """Tokenize a chunk of text (already cut at a safe boundary)."""
        tokens: list[int] = []
        if self.special_tokens:
            sp_pat = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            segments = re.split(sp_pat, chunk)
        else:
            segments = [chunk]
        for seg in segments:
            if seg in self.special_tokens:
                tokens.append(self.special_tokens[seg])
            else:
                for m in re.finditer(self.pattern, seg):
                    m_bytes = list(m.group(0).encode("utf-8"))
                    tokens.extend(self._encode_merge(m_bytes))
        return tokens

    def encode(self, input_path: str, output_path: str) -> None:
        """
            Not really needed for parallelization because of the use of cache (the highest cost).
        """
        assert os.path.isfile(input_path), "Document to encode does not exist"
        leftover = ""
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
             open(output_path, "wb") as f_out:
            while True:
                raw = f_in.read(_ENCODE_CHUNK_BYTES)
                buffer = leftover + raw
                if not buffer:
                    break
                if not raw:
                    # EOF: flush everything remaining
                    np.array(self._tokenize_chunk(buffer), dtype=np.uint16).tofile(f_out)
                    break
                safe_cut = self._last_safe_cut(buffer)
                if safe_cut == 0:
                    # No safe cut yet — read more data
                    leftover = buffer
                    continue
                np.array(self._tokenize_chunk(buffer[:safe_cut]), dtype=np.uint16).tofile(f_out)
                leftover = buffer[safe_cut:]
                    
    def decode(self):
        raise NotImplementedError
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("MyBPE v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merge:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merge.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "MyBPE v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merge = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()
    

def train(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    input_path = Path(input_path)
    file_prefix = input_path.resolve().stem
    pretokenization_path = input_path.resolve().parent / "processed_chunks"

    bpe_tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    bpe_tokenizer.train(
        input_path=input_path,
        pretokenization_path=pretokenization_path,
        file_prefix=file_prefix
    )
    bpe_tokenizer.save("MyModel")

    vocab = bpe_tokenizer.vocab
    merge = [
        (vocab[p0], vocab[p1]) for (p0, p1) in bpe_tokenizer.merge.keys()
    ]
    return vocab, merge


def encode(input_path: str, tokenizer_model_file: str):
    bpe_tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    bpe_tokenizer.load(tokenizer_model_file)

    output_path = str(Path(input_path).stem) + "_encoding.lib"
    bpe_tokenizer.encode(
        input_path=input_path,
        output_path=output_path
    )
    print("Test Encoding Completed.")


if __name__ == "__main__":
    
    input_doc_path = CURRENT_DIR.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1024
    special_tokens = [DOC_SPECIAL_TOKEN.decode("utf-8")]

    tokenizer_model_path = "MyModel.model"
    
    # train(input_doc_path, vocab_size, special_tokens)

    encode_test_input_path = CURRENT_DIR.parent / "data" / "owt_valid.txt"
    encode(str(input_doc_path), tokenizer_model_path)
    pass