import matplotlib.pyplot as plt
import numpy as np
from util import getData, get_xs_to_plot, get_blog_x_data, get_blog_y_data
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
    X = get_blog_x_data('blogdata/x_train.csv')
    y = get_blog_y_data('blogdata/y_train.csv')
    print 'here'
    w_ridge = ridge_regression(X, y, 0.1)
    return w_ridge

def problem_3_1():
    x, y = getData('curvefitting.txt')
    for M in [1, 2, 3]:
        for lam in [0, 0.0003, 0.001, 0.003]:
            w_ridge = polynomial_ridge(x, y, M, lam)
            bishop_plot(x, y, w_ridge, 'ridge regression, M=%s, lam=%s' % (M, lam))
            plt.savefig('problem3_1_M_%s_lam_%s.png' % (M, lam))
            print 'just tried M = %s, lam=%s' % (M, lam)
            print '    sse was %s' % sse(x, y, w_ridge)

def problem_3_2_case(train_x, train_y, valid_x, valid_y, M, lam, A_or_B, plot=False):
    w_ridge = polynomial_ridge(train_x, train_y, M, lam)
    xs_to_plot = get_xs_to_plot(train_x.tolist() + valid_x.tolist())
    y_hat = np.array([compute_yhat(x, w_ridge) for x in xs_to_plot])
    error = sse(valid_x, valid_y, w_ridge)
    if plot:
        plt.close()
        plt.scatter(train_x, train_y, color='blue', label='train ' + A_or_B)
        plt.scatter(valid_x, valid_y, color='red', label='validate')
        plt.plot(xs_to_plot, y_hat, color='black', 
            label='ridge regression, $M=%s$, $\lambda=%s$' % (M, lam))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(loc=4)
        plt.savefig('problem3_2_train%s_M_%s_lam_%s.png' % (A_or_B, M, lam))
        plt.show()
    print 'just tried train=%s, M=%s, lam=%2.2f: sse=%3.4f' % \
        (A_or_B, M, lam, error)
    return error

def problem_3_2(A_or_B, plot_individual=False):
    train_x, train_y = getData('regress%s_train.txt' % A_or_B)
    valid_x, valid_y = getData('regress_validate.txt')
    lams = np.linspace(0, 10, 101)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#808080', '#DDDD00', '#800000']
    best_lams = []
    best_sses = []
    for M in range(10):
        errors = []
        for lam in lams:
            error = problem_3_2_case(train_x, train_y, valid_x, valid_y,
                M, lam, A_or_B, plot_individual)
            errors.append(error)
        if not plot_individual:
            plt.plot(lams, errors, color=colors[M], 
                label='M=%s' % M)
        best_lams.append(np.argmin(errors)/10.)
        best_sses.append(min(errors))
    print best_lams
    print best_sses
    if not plot_individual:
        plt.ylim(0, 30 if A_or_B == 'A' else 100)
        plt.title('Ridge regression error by $M$ and $\lambda$, training set %s' % A_or_B)
        plt.xlabel('$\lambda$')
        plt.ylabel('sum of squared error (against validation data)')
        plt.legend()
        plt.savefig('problem3_2_train%s_meta.png' % A_or_B)
        plt.show()
        plt.close()
        

if __name__ == '__main__':
    #problem_3_2('A')
    #problem_3_2('B')
    problem_3_3()
