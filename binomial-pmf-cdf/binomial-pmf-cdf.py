import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    ans = 0
    for i in range(k+1):
        curr = math.comb(n, i)*(p**i)*((1-p)**(n-i))
        ans += curr
    return {"pmf": float(curr), "cdf": float(ans)}