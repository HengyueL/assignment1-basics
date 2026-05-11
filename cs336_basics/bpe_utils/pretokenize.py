import regex as re
from collections import Counter
from typing import Optional, List

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


if __name__ == "__main__":
    
    test_chunk = "I'm wanting' to i to buy some bananas for bananas 1231233333 today for my wife piggy banana."
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)

    test_chunk = ""
    test_result = pretokenize_single_chunk(test_chunk)
    print(test_result)