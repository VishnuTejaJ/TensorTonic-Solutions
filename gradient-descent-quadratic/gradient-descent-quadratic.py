def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    f = a*x0*x0 + b*x0 + c
    grad = 2*a*x0 + b
    x = x0-lr*grad
    for i in range(1,steps):
        grad = 2*a*x + b
        x = x-lr*grad
    return x