import numpy as np
def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    ans = []
    for i in range(0,len(values)-len(weights)+1):
        num = 0
        for j in range(i,i+len(weights)):
            num += values[j]*weights[j-i]
        ans.append(num/sum(weights))
    return ans