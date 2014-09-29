import numpy as np
import math
import time
import signal
from grad_descent_plot_1 import grad_descent_plot_2D as gdplt

def gradient_descent(f, guess, step=0.1, thresh=1e-10, g=None, guess_list=None,
        halt=True):
    # f: scalar function of vector argument to optimize
    # guess: initial vector guess of the minimum
    # step: step size
    # thresh: converges when difference in f value on two successive steps below thresh
    # g: gradient of f; default is numerical estimate
    # halt: whether to stop after 1000 guesses
    if not g:
        g = gradient(f)
    if guess_list is None:
        guess_list = []
    grad_init = g(guess)
    new_guess = guess - grad_init*step
    guess_list += [guess, new_guess]
    while abs(f(guess)-f(new_guess)) >= thresh:
        guess = new_guess
        grad = g(new_guess)
        new_guess = guess - grad*step
        guess_list.append(new_guess)
        if len(guess_list) > 1000 and halt:
            print '    WARNING did not complete'
            break
    return new_guess

def gradient_descent_backtrack(f, guess, thresh=1e-10, beta=0.5, g=None):
    # f: scalar function of vector argument to optimize
    # guess: initial vector guess of the minimum
    # thresh: converges when difference in f value on two successive steps below thresh
    # beta: governs backtracking, between 0 and 1
    # g: gradient of f; default is numerical estimate
    if not g:
        g = gradient(f)
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

def gradient(f, h=1e-10):
    # f: scalar function of a vector
    # h: step size
    def g(x):
        # x: center point
        dim = len(x)
        newx = [0]*dim 
        for i in range(dim):
            deriv_dim = np.zeros(dim)
            deriv_dim[i] = 1
            newx[i] = (f(x+h*deriv_dim) - f(x-h*deriv_dim))/float(2*h)
        return np.array(newx)
    return g

def example_func_1(x):
    """The example function from the book."""
    # x: a two-dimensional horizontal vector
    return 0.5*(x[0]**2 - x[1])**2 + 0.5*(x[0] - 1)**2

def example_func_1_grad(x):
    # x: a two-dimensional horizontal vector
    d0 = (x[0]**2 - x[1])*2*x[0] + (x[0] - 1)
    d1 = x[1] - x[0]**2
    return np.array([d0, d1])

def example_func_2(x):
    """A convex function."""
    # x: a two-dimensional horizontal vector
    return x[0]**4 + x[1]**4

def example_func_2_grad(x):
    # x: a two-dimensional horizontal vector
    return np.array([4*(x[0] - 7)**3, 4*(x[1] - 8.1)**3])

def example_func_3(x):
    """Ackley's function"""
    # x: a two-dimensional horizontal vector
    return - 20*math.exp(-0.2*(0.5*(x[0]**2 + x[1]**2))**0.5) \
               - math.exp(0.5*(math.cos(2*math.pi*x[0]) + math.cos(2*math.pi*x[1]))) \
               + 20 + math.e

def example_func_4(x):
    """Easom function"""
    # x: a two-dimensional horizontal vector
    return - math.cos(x[0]) * math.cos(x[1]) * \
        math.exp(-(x[0]-math.pi)**2 -(x[1]-math.pi)**2)

if __name__ == "__main__":

    def timeout_handler(signum, frame):
        raise Exception("  TIMEOUT")

    def go(f, guess, step, thresh, name):
        print 'finding minimum of', name
        print '  with guess', guess, 'step', step, 'thresh', thresh
        # Halt computation after 5 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        try:
            guess_list = []
            x = gradient_descent(
                    f, 
                    np.array(guess), 
                    step=step, 
                    thresh=thresh,
                    guess_list=guess_list)
        except Exception, m:
            print m
            return
        signal.alarm(0)
        print '  minimum is', f(x), 'attained at x = ', x
        X0 = np.array([x[0] for x in guess_list])
        X1 = np.array([x[1] for x in guess_list])
        TrueMin = np.array([0, 0])
        file_name = name + '_' + str(guess[0]) + '_' + str(guess[1]) +\
            '_' + str(step) + '_' + str(thresh)
        gdplt(f, X0, X1, TrueMin, file_name, lim=(3 if name=='convex' else 0))

    go(example_func_2, [2,1.5], 0.04, 1e-10, 'convex')
    go(example_func_2, [2,1.5], 0.04, 1e-3, 'convex')
    go(example_func_2, [2,1.5], 0.1, 1e-10, 'convex')
    go(example_func_2, [2.3,2], 0.1, 1e-10, 'convex')
    go(example_func_2, [2,1.5], 0.13, 1e-10, 'convex')
    go(example_func_3, [0.5,0.6], 0.005, 1e-10, 'bumpy')
    go(example_func_3, [0.5,0.6], 0.03, 1e-10, 'bumpy')
    go(example_func_3, [2.5,2.6], 0.005, 1e-10, 'bumpy')
    go(example_func_3, [2.5,2.6], 0.1, 1e-10, 'bumpy')

