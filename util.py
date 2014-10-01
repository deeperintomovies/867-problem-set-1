import numpy as np

def getData(name):
    data = np.loadtxt(name)
    # Returns column vectors
    X = data[0:1].T
    Y = data[1:2].T
    return X.ravel(), Y.ravel()


def get_xs_to_plot(values):
    """Takes some x values and returns a ndarray of x values spaced evenly
    through a range slightly wider than that of the given values"""
    xmin = min(values) 
    xmax = max(values) 
    span = xmax - xmin
    xmin -= 0.02*span
    xmax += 0.02*span
    return np.linspace(xmin, xmax, 400)

