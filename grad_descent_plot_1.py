import matplotlib.pyplot as plt
import numpy as np


def grad_descent_plot_2D(f, X0, X1, TrueMin, OutName='grad-descent-example'):
# Plot route of gradient descent.
    plt.figure(1)
    print X0, X1, X0.shape, X1.shape
    plt.plot(X0, X1, 'o-', ms=6, lw=3, color='blue')
    plt.plot([TrueMin[0]], [TrueMin[1]], 'x', ms=12, markeredgewidth=3,
          color='orange')
    min_x0 = min(min(X0), TrueMin[0])
    max_x0 = max(max(X0), TrueMin[0])
    min_x1 = min(min(X1), TrueMin[1])
    max_x1 = max(max(X1), TrueMin[1])
    xlim_min, xlim_max = min_x0-0.1*abs(min_x0-max_x0), max_x0+0.1*abs(min_x0-max_x0)
    plt.xlim(xlim_min, xlim_max)
    ylim_min, ylim_max = min_x1-0.1*abs(min_x1-max_x1), max_x1+0.1*abs(min_x1-max_x1)
    plt.ylim(ylim_min, ylim_max)
    delta_x = len(X0)/float(2*abs(xlim_max-xlim_min))
    delta_y = len(X1)/float(2*abs(ylim_max-ylim_min))
    x_points = np.arange(xlim_min, xlim_max, delta_x)
    y_points = np.arange(ylim_min, ylim_max, delta_y)
    Xgrid, Ygrid = np.meshgrid(x_points, y_points)
    Z = np.zeros(Xgrid.shape)
    for i in range(Xgrid.shape[0]):
        for j in range(Xgrid.shape[1]):
            Z[i][j] = f(np.array([Xgrid[i][j], Ygrid[i][j]]))
    plt.contour(Xgrid, Ygrid, Z)
    plt.xlabel(r'$x0$', fontsize=24)
    plt.ylabel(r'$x1$', fontsize=24)
    plt.savefig(OutName+'.png')
    plt.show()
