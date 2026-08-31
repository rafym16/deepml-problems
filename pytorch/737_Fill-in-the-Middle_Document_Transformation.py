def fim_transform(tokens: list, i: int, j: int) -> list:
    """
    Apply Fill-in-the-Middle (PSM) transformation to a token sequence.

    Args:
        tokens: list of token strings representing the document
        i: index where prefix ends / middle begins
        j: index where middle ends / suffix begins

    Returns:
        Transformed list of tokens in PSM format with special tokens.
    """
    # Your code here
    prefix = tokens[0:i]
    suffix = tokens[j:]
    mid = tokens[i:j]

    prefix.insert(0, '<PRE>')
    suffix.insert(0, '<SUF>')
    mid.insert(0, '<MID>')
    result = prefix + suffix + mid
    result.append('<EOT>')
    return result


