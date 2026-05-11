import os, sys
from typing import BinaryIO, Optional, Tuple
import json
from pathlib import Path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from bpe_utils.pretokenize import pretokenize_single_chunk
from multiprocessing import Pool

N_POOL = 8
DOC_SPECIAL_TOKEN = b"<|endoftext|>"


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
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    pretoken_count_dict = pretokenize_single_chunk(chunk, special_token_list=[DOC_SPECIAL_TOKEN])
    save_file_path = save_dir / f"{file_name}_{i_chunk:06d}.json"
    with open(save_file_path, "w") as out:
        json.dump(pretoken_count_dict, out)


def main(file_path: Path, save_dir: Optional[Path] = None):
    file_name = file_path.name
    if not save_dir:
        save_dir = current_dir / "tmp"
    os.makedirs(save_dir, exist_ok=True)

    ## Usage
    with open(file_path, "rb") as f:
        num_processes = 64
        boundaries = find_chunk_boundaries(f, num_processes, DOC_SPECIAL_TOKEN)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # for i_chunk, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # # Run pre-tokenization on your chunk and store the counts for each pre-token
            # pretoken_count_dict = pretokenize_single_chunk(chunk)
            # save_file_path = save_dir / f"file_name_{i_chunk:06d}.json"
            # with open(save_file_path, "w") as out:
            #     json.dump(pretoken_count_dict, out)
        
        # Parallel implementation
        task = [
            (file_path, start, end, save_dir, i_chunk, file_name)
            for i_chunk, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]
        with Pool(N_POOL) as p:
            p.map(process_chunk, task)
        
        print("Completed.")


if __name__ == "__main__":
    test_file_path = Path(__file__).resolve().parent.parent / "data" / "owt_valid.txt"
    # file_name = test_file_path.name
    # print(file_name)
    # print(type(test_file_path))
    main(test_file_path)

