import os, sys
import regex as re
from collections import Counter
from typing import Optional, List, BinaryIO, Tuple
from pathlib import Path
import json
from multiprocessing import Pool
import psutil
import logging
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
N_POOL = os.cpu_count()
RAM_MB = psutil.virtual_memory().total // (1024**2)


def pretokenize_single_chunk(
    text_chunk: str, 
    special_token_list: Optional[List[str]] = None,
    pattern: str = GPT4_SPLIT_PATTERN
) -> Counter[str]:
    """
        Pretokenize a single text chunk. 
        Output a dictionary of {pretokens: count}.
    """
    if special_token_list:
        for special_token in special_token_list:
            text_chunk = text_chunk.replace(special_token, " ")
    
    chunk_iter = re.finditer(pattern, text_chunk)  # Pretokenize iter
    
    return_dict = {}
    for m in chunk_iter:
        pretoken = m.group()
        return_dict[pretoken] = return_dict.get(pretoken, 0) + 1
    
    return Counter(return_dict)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args: Tuple):
    
    file_path, start, end, save_dir, i_chunk, file_name, special_token_list = args
    
    save_file_path = save_dir / f"{file_name}_{i_chunk:06d}.json"
    if os.path.isfile(save_file_path):
        print(f"File ({save_file_path}) exists. Skip processing chunk")
        return
    
    logger.info(f"Processing {file_path} - Chunk {i_chunk}")

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    pretoken_count_dict = pretokenize_single_chunk(chunk, special_token_list=special_token_list)

    with open(save_file_path, "w") as out:
        json.dump(pretoken_count_dict, out)
    logger.info(f"File saved to: {save_file_path}")


def pre_tokenize_document(
    input_path: str, 
    output_path: str,
    document_split_bytes: bytes,
    special_tokens_list: List[str]
):
    assert os.path.isfile(input_path), "Raw document does not exist"

    document_path = Path(input_path)
    file_prefix = str(Path(document_path).stem)
    logger.info(f"File prefix: {file_prefix}")

    save_dir = Path(output_path)
    os.makedirs(save_dir, exist_ok=True)

    # Process one document
    with open(document_path, "rb") as file:
        logger.info(f"Pretokenize file: {document_path}")

        file_size = Path(document_path).stat().st_size
        target_chunk_size = min(512, RAM_MB // (4 * 4 * N_POOL)) * 2 ** 20
        num_processes = max(1, min(N_POOL * 2, file_size // target_chunk_size))

        logger.info(
            f"File size: {file_size / 1024**2 :.02f} mb - "
            f"Chunk size: {target_chunk_size / 2**20 :.02f} mb - "
            f"N Chunks: {num_processes}"
        )

        boundaries = find_chunk_boundaries(file, num_processes, document_split_bytes)

        task = [
            (document_path, start, end, save_dir, i_chunk, file_prefix, special_tokens_list)
            for i_chunk, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]

        with Pool(N_POOL) as p:
            p.map(process_chunk, task)
    
    logger.info(f"{document_path} has been successfully pre-tokenized.")


if __name__ == "__main__":
    
    test_chunk = "I'm wanting' to i to buy some bananas for bananas 1231233333 today for my wife piggy banana."
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)

    test_chunk = ""
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)