import re
import torch

def unigram_probability(corpus: str, word: str) -> torch.Tensor:
    tokens = re.findall(r'<s>|</s>|\w+', corpus)

    if not tokens:
        return torch.tensor(0.0)

    match_mask = torch.tensor([t == word for t in tokens], dtype=torch.float32)

    word_count = torch.sum(match_mask)
    total_tokens = torch.tensor(len(tokens), dtype=torch.float32)

    prob = word_count / total_tokens
    return prob

