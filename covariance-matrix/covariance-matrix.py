import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    n = len(X)
    X = np.array(X)
    X = X - np.mean(X, axis=0)
    return (X.T@X)/(n-1)
    pass