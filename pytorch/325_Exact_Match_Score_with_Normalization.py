import string
import torch

def exact_match_score(predictions: list[str], references: list[str]) -> torch.Tensor:
    """
    Calculate the exact match score between predictions and references.

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        Exact match score as a float between 0 and 1
    """

    # Your code here
    def normalize(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return "".join(text.split())

    if not predictions or not references or len(predictions) != len(references):
        return torch.tensor([0.0])

    normalized_pred = [normalize(p) for p in predictions]
    normalized_ref = [normalize(r) for r in references]
    matches = sum(p == r for p, r in zip(normalized_pred, normalized_ref))
    tensor_match = torch.tensor(matches)
    return tensor_match / len(predictions)

predictions = ['Hello, World!', 'The answer is 42']
references = ['hello world', 'the answer is 42']

print(exact_match_score(predictions, references))