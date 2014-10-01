import matplotlib.pyplot as plt
import numpy as np
from util import getData, get_xs_to_plot
from p2 import bishop_plot, sse, compute_yhat

def polynomial_ridge(x, y, M, l):
    # x: data x coords
    # y: data y coords
    # M: fit polynomial of degree M
    # l: lambda in ridge regression
    # return: ridge regression coeffs
    X = np.array([[i**d for d in range(M+1)] for i in x])
    return ridge_regression(X, y, l)

def ridge_regression(X, y, lam):
    """General ridge regression (not just for polynomial fitting)
    
    X: N-by-D ndarray 
    y: N ndarray
    lam: lambda
    return: ridge regression coeffs
    """
    D = X.shape[1]
    return np.linalg.inv(lam*np.identity(D) + X.T.dot(X)).dot(X.T).dot(y)

def problem_3_3():
    pass

def problem_3_1():
    x, y = getData('curvefitting.txt')
    for M in [3, 4, 5, 6, 7, 8, 9]:
        for lam in [0, 0.0003, 0.001, 0.003]:
            w_ridge = max_likelihood_ridge(x, y, M, lam)
            bishop_plot(x, y, w_ridge, 'ridge regression, M=%s, lam=%s' % (M, lam))
            plt.savefig('problem3_1_M_%s_lam_%s.png' % (M, lam))
            print 'just tried M = %s, lam=%s' % (M, lam)
            print '    sse was %s' % sse(x, y, w_ridge)

def problem_3_2(A_or_B, plot=False):
    train_x, train_y = getData('regress%s_train.txt' % A_or_B)
    valid_x, valid_y = getData('regress_validate.txt')
    for M in range(10):
        for lam in [0, 0.1, 0.5, 1, 5, 10]:
            w_ridge = max_likelihood_ridge(train_x, train_y, M, lam)
            xs_to_plot = get_xs_to_plot(train_x.tolist() + valid_x.tolist())
            y_hat = np.array([compute_yhat(x, w_ridge) for x in xs_to_plot])
            if plot:
                plt.close()
                plt.scatter(train_x, train_y, color='blue', label='train ' + A_or_B)
                plt.scatter(valid_x, valid_y, color='red', label='validate')
                plt.plot(xs_to_plot, y_hat, color='black', 
                    label='ridge on train, M=%s, lam=%s' % (M, lam))
                plt.legend()
                plt.savefig('problem3_2_train%s_M_%s_lam_%s.png' % (A_or_B, M, lam))
            print 'just tried train=%s, M=%s, lam=%2.2f: sse=%3.4f' % \
                (A_or_B, M, lam, sse(valid_x, valid_y, w_ridge))

if __name__ == '__main__':
    problem_3_2('A')
    problem_3_2('B')
