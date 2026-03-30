def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    ans = {}
    for sentence in sentences:
        for word in sentence:
            if word in ans:
                ans[word] += 1
            else:
                ans[word] = 1
    return ans