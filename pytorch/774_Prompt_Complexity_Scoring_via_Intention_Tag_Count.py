def score_prompt_complexity(prompts):
    """
    prompts: list of {'text': str, 'tags': list[str]}
    Returns: list of {'text': str, 'complexity': int} sorted by complexity desc (stable).
    """
    prompts.sort(key=lambda x: x['tags'], reverse=True)
    result = []
    for prompt in prompts:
        complexities = {
            'text': prompt['text'],
            'complexity': len(set(prompt['tags']))
        }
        result.append(complexities)

    return result


