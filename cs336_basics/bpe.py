"""
    Implement the bpe tokenizer.
"""
import sys, os
import psutil
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import logging
logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool
from bpe_utils.pretokenize import (
    process_chunk,
    find_chunk_boundaries,
    DOC_SPECIAL_TOKEN
)

RAW_DOCUMENT_DIR = CURRENT_DIR.parent / "data"
N_POOL = os.cpu_count() - 1
RAM_MB = psutil.virtual_memory().total // (1024**2)


class BPETokenizer:
    """
        BPE tokenizer class
    """
    def __init__(self):
        # Merge rules: (int, int) -> int
        self.merge = {}

        # str -> int, e.g. {'<|endoftext|>': 100257}, stored from the end of total vocab
        self.special_tokens = {}
        self.inverse_special_tokens = {}  # Inverse mapping of special tokens

        # default: vocab size of 256 (all bytes), no merges, no patterns
        # {idx: __repr__()}
        self.vocab = self._build_vocab()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RAM: {RAM_MB} mb")
    
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

    def _pre_tokenize_dataset(self):
        assert os.path.isdir(RAW_DOCUMENT_DIR), "Raw document does not exist"

        raw_document_list = [f for f in os.listdir(RAW_DOCUMENT_DIR) if f.endswith(".txt")]
        self.logger.info(f"Found the following raw document: {raw_document_list}")

        for _, document_name in enumerate(raw_document_list):
            document_path = RAW_DOCUMENT_DIR / f"{document_name}"
            file_prefix = str(Path(document_name).stem)
            save_dir = RAW_DOCUMENT_DIR / "processed_chunks" / f"{file_prefix}"
            os.makedirs(save_dir, exist_ok=True)

            # Process one document
            with open(document_path, "rb") as file:
                self.logger.info(f"Pretokenize file: {document_path}")

                file_size = Path(document_path).stat().st_size
                target_chunk_size = min(512, RAM_MB // (4*4)) * 2 ** 20 # 256 MB chunk
                num_processes = max(N_POOL * 2, file_size // target_chunk_size)
                self.logger.info(
                    f"File size: {file_size / 1024**2 :.02f} mb - "
                    f"Chunk size: {target_chunk_size / 2**20 :.02f} mb - "
                    f"N Chunks: {num_processes}"
                )

                boundaries = find_chunk_boundaries(file, num_processes, DOC_SPECIAL_TOKEN)

                task = [
                    (document_path, start, end, save_dir, i_chunk, file_prefix)
                    for i_chunk, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
                ]

                with Pool(N_POOL) as p:
                    p.map(process_chunk, task)
            
            self.logger.info(f"{document_name} has been successfully pre-tokenized.")

    def train(self):
        raise NotImplementedError


if __name__ == "__main__":
    
    bpe_tokenizer = BPETokenizer()

    # Test pre_tokenize
    bpe_tokenizer._pre_tokenize_dataset()
    pass