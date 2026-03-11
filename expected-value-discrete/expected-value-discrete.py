import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    e = 0
    if sum(p)!=1:
        raise ValueError
    for i in range(len(x)):
        e += x[i]*p[i]
    return e