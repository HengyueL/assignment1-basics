from typing import Tuple, Optional, Dict

def get_stats(
    input_list: Tuple[int], 
    multiplier: int, 
    counts: Optional[Dict]
):
    """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
    """

    counts = {} if not counts else counts
    for p0, p1 in zip(input_list[:-1], input_list[1:]):
        pair = (p0, p1)
        counts[pair] = counts.get(pair, 0) + multiplier
    return counts


def merge_key(key: Tuple, pair: Tuple, idx: int):
    ...