import re
import numpy as np


def unigram_probability(corpus: str, word: str) -> float:
    tokens = re.findall(r'<s>|</s>|\w+', corpus)
    corpus_array = np.array(tokens)

    total_tokens = len(corpus_array)

    if total_tokens == 0:
        return 0.0

    word_count = np.sum(corpus_array == word)

    prob = float(word_count / total_tokens)

    return prob