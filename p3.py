import matplotlib.pyplot as plt
import numpy as np
from util import getData

def max_likelihood_ridge(x, y, M, l):
    # x: data x coords
    # y: data y coords
    # M: fit polynomial of order M
    # l: lambda in ridge regression
    # w_ridge: ridge regression coeffs
    lambda_I = l*np.identity(M+1)
    X = np.array([[i**d for d in range(M+1)] for i in x])
    X_square = (X.T.dot(X))
    X_inv = np.linalg.inv(lambda_I + X_square)
    X_final = X_inv.dot(X.T)
    w_ridge = X_final.dot(y)
    return w_ridge

