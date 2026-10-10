import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.array(x)
    pmf = np.where(x == 1, p, 1.0 - p).astype(float)
    return {"pmf": pmf, "mean": float(p), "variance": float(p*(1-p))}