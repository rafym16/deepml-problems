def dup_ngram_ratio(text: str, n: int) -> float:
    # Your code here
    splitted_text = text.lower().split()
    list_gram = [tuple(splitted_text[i:i + n]) for i in range(len(splitted_text) - n + 1)]

    if not list_gram:
        return 0.0

    ngram_counts = {}
    for ngram in list_gram:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    duplicated_count = sum(count for count in ngram_counts.values() if count > 1)

    ratio = round((duplicated_count / len(list_gram)), 4)
    return ratio