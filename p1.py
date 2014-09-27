import numpy as np

def gradient_descent(f, g, guess, step, thresh):
    # f: scalar function of vector argument to optimize
    # g: gradient computation
    # guess: initial vector guess of the minimum
    # step: step size
    # thresh: converges when difference in f value on two successive steps below thresh
    grad_init = g(guess)
    new_guess = guess - grad_init*step
    while abs(f(guess)-f(new_guess)) >= thresh:
        guess = new_guess
        grad = g(new_guess)
        new_guess = guess - grad*step
    return new_guess

def gradient_descent_backtrack(f, g, guess, thresh, beta=0.5):
    # f: scalar function of vector argument to optimize
    # g: gradient computation
    # guess: initial vector guess of the minimum
    # thresh: converges when difference in f value on two successive steps below thresh
    # beta: governs backtracking, between 0 and 1
    while True:
        grad = g(guess) 
        step = 1 
        while f(guess - step*grad) > f(guess) - 0.2 * step * np.linalg.norm(grad)**2:
            step *= beta
        new_guess = guess - step*grad
        if abs(f(guess) - f(new_guess)) < thresh:
           break 
        guess = new_guess
    return guess

def gradient(f, x, h):
    # f: scalar function of a vector
    # x: center point
    # h: step size
    
    dim = len(x)
    newx = [0]*dim 
    for i in range(dim):
        deriv_dim = np.zeros(dim)
        deriv_dim[i] = 1
        newx[i] = (f(x+h*deriv_dim) - f(x-h*deriv_dim))/float(2*h)
    return np.array(newx)

def example_func_1(x):
    # x a horizontal vector
    return 0.5*(x[0]**2 - x[1])**2 + 0.5*(x[0] - 1)**2

def example_func_1_grad(x):
    # x a horizontal vector
    d0 = (x[0]**2 - x[1])*2*x[0] + (x[0] - 1)
    d1 = x[1] - x[0]**2
    return np.array([d0, d1])
