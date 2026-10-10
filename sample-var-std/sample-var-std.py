import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x_mean = np.mean(x)
    var = 0
    for i in x:
        var += (i-x_mean)**2
    var /= (len(x)-1)
    return {"variance": float(var), "standard_deviation": float(var**0.5)}