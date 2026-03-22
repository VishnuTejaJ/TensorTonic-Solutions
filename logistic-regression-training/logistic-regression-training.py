import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros(*X[0].shape)
    b = 0
    loss = 0.0
    # print(np.dot(w.T,X[0])+b)
    for step in range(1,steps+1):
        for i,x in enumerate(X):
            z = np.dot(w.T,x)+b
            # print(z)
            p = _sigmoid(z)
            # print(p)
            dL_dw = (p-y[i])*x
            # print(y[i])
            # print(dL_dw)
            dL_db = p-y[i]
            # print(dL_db)
            w -= lr*dL_dw
            b -= lr*dL_db
            loss += y*np.log(p)+(1-y)*np.log(1-p)
            if step%10==0:
                print(loss/step)
    return w,b