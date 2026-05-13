import os, sys
import regex as re
from collections import Counter
from typing import Optional, List, BinaryIO, Tuple
from pathlib import Path
import json
import logging
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

DOC_SPECIAL_TOKEN = b"<|endoftext|>"
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


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
            text_chunk = text_chunk.replace(special_token.decode("utf-8"), "")
    
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
    
    file_path, start, end, save_dir, i_chunk, file_name = args
    
    save_file_path = save_dir / f"{file_name}_{i_chunk:06d}.json"
    if os.path.isfile(save_file_path):
        print(f"File ({save_file_path}) exists. Skip processing chunk")
        return
    
    logger.info(f"Processing {file_path} - Chunk {i_chunk}")

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    pretoken_count_dict = pretokenize_single_chunk(chunk, special_token_list=[DOC_SPECIAL_TOKEN])

    with open(save_file_path, "w") as out:
        json.dump(pretoken_count_dict, out)
    logger.info(f"File saved to: {save_file_path}")


if __name__ == "__main__":
    
    test_chunk = "I'm wanting' to i to buy some bananas for bananas 1231233333 today for my wife piggy banana."
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)

    test_chunk = ""
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)