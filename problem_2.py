import matplotlib.pyplot as plt
import numpy as np

def getData(name):
    data = np.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def max_likelihood(x, y, M):
    # x: np.array of x coords of data
    # y: np.array of y coords of data
    # M: fit polynomial of order M
    # w: regression coeffs---ie, y_ols(x) = w_ols[0] + w_ols[1]*x +...+ w_ols[M]*x^M
    X = np.array([[i**d for d in range(M+1)] for i in x]).reshape((len(x), M+1))
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



def sse_derivative(x, y, w):
    # x: data x coords
    # y: data y coords
    # w: regression coeffs choice---y_hat(x) = w[0]*x**0+...+w[M]*x**M
    # returns gradient vector [d/dw_k sse(x, y, w) for k in range(len(w))]

    error_vector = np.array([y[i]-compute_yhat(x[i], w) for i in range(len(x))])
    coeffs = np.array([[-2*k*w[k]*x[i]**(k-1) for i in range(len(x))] for k in range(len(w))])
    return coeffs.dot(error_vector)


def problem_2(M):
    x,y = getData('curvefitting.txt')
    w_ols = max_likelihood(x, y, M)
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
