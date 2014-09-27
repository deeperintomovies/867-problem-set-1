import numpy as np
import math
import time
import signal

def gradient_descent(f, guess, step=0.1, thresh=1e-10, g=None):
    # f: scalar function of vector argument to optimize
    # guess: initial vector guess of the minimum
    # step: step size
    # thresh: converges when difference in f value on two successive steps below thresh
    # g: gradient of f; default is numerical estimate
    if not g:
        g = gradient(f)
    grad_init = g(guess)
    new_guess = guess - grad_init*step
    while abs(f(guess)-f(new_guess)) >= thresh:
        guess = new_guess
        grad = g(new_guess)
        new_guess = guess - grad*step
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
    return (x[0] - 7)**2 + (x[1] - 8.1)**2

def example_func_2_grad(x):
    # x: a two-dimensional horizontal vector
    return np.array([2*(x[0] - 7), 2*(x[1] - 8.1)])

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
    print "here"

    convex_guesses = [[0,0], [3,2], [7,8], [1e6, 1e6]]
    convex_steps = [1e-3, 1e-2, 1e-1, 1e0, 1e1] 
    convex_threshes = [1e-10, 1e-5, 1e-1]

    bumpy_guesses = [[0,0], [0.5,0], [0.5,0.5], [1,1], [1,1.5]]
    bumpy_steps = convex_steps
    bumpy_threshes = convex_threshes

    def timeout_handler(signum, frame):
        raise Exception("  TIMEOUT")

    def go(f, guesses, steps, threshes, name):
        for guess in guesses:
            for step in steps:
                for thresh in threshes:
                    print 'finding minimum of', name
                    print '  with guess', guess, 'step', step, 'thresh', thresh
                    # Halt computation after 2 seconds
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(2)
                    try:
                        x = gradient_descent(f, np.array(guess), step=step, thresh=thresh)
                        print '  minimum is', f(x), 'attained at x = ', x
                    except Exception, m:
                        print m
                    signal.alarm(0)

    go(example_func_2, convex_guesses, convex_steps, convex_threshes, 'convex example f')
    go(example_func_3, bumpy_guesses, bumpy_steps, bumpy_threshes, 'bumpy example f')

