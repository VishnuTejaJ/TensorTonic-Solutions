import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x = np.array(x)
    y = np.array(y)
    # print(x[i]*y[i] for i in range(len(x)))
    return float(sum(x*y))