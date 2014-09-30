import matplotlib.pyplot as plt
import numpy as np
from util import getData

def max_likelihood(x, y, M):
    # x: data x coords
    # y: data y coords
    # M: fit polynomial of order M
    # w_ols: ols regression coeffs---ie,
        # y_ols(x) = w_ols[0] + w_ols[1]*x +...+ w_ols[M]*x^M
    X = np.array([[i**d for d in range(M+1)] for i in x])#.reshape((len(x), M+1))
    X_square = (X.T.dot(X))
    X_inv = np.linalg.inv(X_square)
    X_final = X_inv.dot(X.T)
    w_ols = X_final.dot(y)
    return w_ols

def sse(x, y, w):
    # x: data x coords
    # y: data y coords
    # w: regression coeffs choice---y_hat(x) = w[0]*x**0+...+w[M]*x**M
    # returns sum of squared errors between ys and y_hats
    err = 0
    for i,j in zip(x,y):
        y_hat = compute_yhat(i, w)
        err += (j-y_hat)**2
    return err

def compute_yhat(x_0,w):
    # x_0: single x coord
    # w: regression coeffs choice---y_hat(x) = w[0]*x**0+...+w[M]*x**M
    # returns y_hat

    return sum([w[exp]*(x_0**exp) for exp in range(len(w))])

def get_sse_and_sse_prime(x,y):
    def f(w):
        return sse(x,y,w)
    def fprime(w):
        return sse_derivative(x,y,w)
    return (f, fprime)

def sse_derivative(x, y, w):
    # x: data x coords
    # y: data y coords
    # w: regression coeffs choice---y_hat(x) = w[0]*x**0+...+w[M]*x**M
    # returns gradient vector [d/dw_k sse(x, y, w) for k in range(len(w))]

    residual_vector = np.array([y[i]-compute_yhat(x[i], w) for i in range(len(x))])
    coeffs = np.array([[-2*x[i]**(k) for i in range(len(x))] for k in range(len(w))])
    return coeffs.dot(residual_vector)


def problem_2(M):
    x,y = getData('curvefitting.txt')
    w_ols = max_likelihood(x, y, M)
    print w_ols
    low_limit_x = min(x) - 0.1*abs(max(x)-min(x)) 
    hi_limit_x = max(x) + 0.1*abs(max(x)-min(x))
    x_points = np.linspace(low_limit_x, hi_limit_x, 100)
    y_hat = np.array([compute_yhat(pt, w_ols) for pt in x_points])
    low_limit_y = min(y) - 0.2*abs(max(y) - min(y))
    hi_limit_y = max(y) + 0.2*abs(max(y)-min(y))
    plt.plot(x_points, y_hat, '-b', label='Max Likelihood, M='+str(M))
    plt.scatter(x, y, label='Data')
    plt.plot(x_points, np.sin(2*np.pi*x_points), '-', color='0.5', label='Sin(2*pi*x)')
    plt.xlim(low_limit_x, hi_limit_x)
    plt.ylim(low_limit_y, hi_limit_y)
    plt.legend()
    plt.show()
    plt.savefig('Problem2_M_'+str(M)+'.png')    




def bishop_plot(x, y, w, label):
    # l: a string to be used in the filename
    low_limit_x = min(x) - 0.1*abs(max(x)-min(x)) 
    hi_limit_x = max(x) + 0.1*abs(max(x)-min(x))
    x_points = np.linspace(low_limit_x, hi_limit_x, 100)
    y_hat = np.array([compute_yhat(pt, w) for pt in x_points])
    low_limit_y = min(y) - 0.2*abs(max(y) - min(y))
    hi_limit_y = max(y) + 0.2*abs(max(y)-min(y))
    plt.plot(x_points, y_hat, '-b', label=label)
    plt.scatter(x, y, label='Data')
    plt.plot(x_points, np.sin(2*np.pi*x_points), '-', color='0.5', label='Sin(2*pi*x)')
    plt.xlim(low_limit_x, hi_limit_x)
    plt.ylim(low_limit_y, hi_limit_y)
    plt.legend()
    plt.show()

def problem_2_1(max_M=15):
    sse_M = []
    l_2_norm = []
    for i in range(max_M): 
        x,y = getData('curvefitting.txt')
        x = x.ravel()
        y = y.ravel()
        w_ols = max_likelihood(x,y,i)
        bishop_plot(x,y,w_ols,'maximum likelihood, M = %s' % i)
        plt.savefig('problem2_ml_M_%s.png' % i)
        sse_M.append(sse(x,y,w_ols))
        l_2_norm.append(sum([ii**2 for ii in w_ols])**0.5)
    plt.close()
    plt.plot(np.array(range(max_M)), np.array(sse_M), label='Sum of Squared Errors')
    plt.savefig('problem2_OLS_SSEvsM.png')
    plt.close()
    plt.plot(np.array(range(max_M)), np.array(l_2_norm), label='L2 Norm of Weight Vector')
    plt.savefig('problem2_OLS_L2_NormvsM.png')
    plt.close()
    x,y = getData('curvefitting.txt')
    x = x.ravel()
    y = y.ravel()
    indices = np.random.choice(max(x.size, y.size)+1, 2)
    for ii in np.sort(indices)[::-1]:
        x = np.delete(x,ii)
        y = np.delete(y,ii)
    for M in [0,1,3,9]:
        w_ols = max_likelihood(x.ravel(),y.ravel(),M)
        bishop_plot(x,y,w_ols,'maximum likelihood w/missing, M = %s' % M)
        plt.savefig('problem2_mlmissing_M_%s.png' % M)

if __name__ == '__main__':
    problem_2_1()
