import os, time, math, torch
import numpy as np
from pathlib import Path


def open_memmap_1d(
    file_path: str | os.PathLike | Path,
    np_dtype: str = "uint16"
) -> np.memmap:
    """
        Open a 1D token memmap file. The file is assumed to be a raw binary array.
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(file_path)
    
    if nbytes % itemsize != 0:
        raise ValueError(f"File size not divisible by dtype size: {file_path} ({nbytes}, itemsize={itemsize})")
    length = nbytes // itemsize
    
    return np.memmap(file_path, mode="r", dtype=dtype, shape=(length, ))