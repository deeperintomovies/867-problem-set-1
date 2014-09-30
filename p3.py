import matplotlib.pyplot as plt
import numpy as np
from util import getData
from p2 import bishop_plot, compute_yhat

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


def problem_3_1():
    x, y = getData('curvefitting.txt')
    x = x.ravel()
    y = y.ravel()
    for M in [3, 4, 5, 7, 9]:
        for lam in [0, 0.001, 0.01, 0.1]:
            w_ridge = max_likelihood_ridge(x, y, M, lam)
            bishop_plot(x, y, w_ridge, 'ridge regression, M=%s, lam=%s' % (M, lam))
            plt.savefig('problem3_1_M_%s_lam_%s.png' % (M, lam))

if __name__ == '__main__':
    problem_3_1()
