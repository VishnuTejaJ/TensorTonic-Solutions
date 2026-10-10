from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    n = len(x)
    x.sort()
    if n%2==0:
        median = (x[n//2]+x[(n//2)-1])/2
    else:
        median = x[n//2]
    mode = x[0]
    mean = x[0]
    curr_freq = 1
    most_freq = 1
    for i in range(1,n):
        mean += x[i]
        if x[i-1]==x[i]:
            curr_freq += 1
        else:
            curr_freq = 1
        if curr_freq>most_freq:
            most_freq = curr_freq
            mode = x[i]
    mean = mean/n
    return {"mean":float(mean),"median":float(median),"mode":float(mode)}