import matplotlib.pyplot as plt
import numpy as np


def grad_descent_plot_2D(f, X0, X1, TrueMin, OutName='grad-descent-example', lim=0):
# Plot route of gradient descent.
    plt.figure(1)
    plt.plot([TrueMin[0]], [TrueMin[1]], 'x', ms=12, markeredgewidth=3,
          color='orange')
    plt.plot(X0[9:200], X1[9:200], 'o-', ms=2, lw=1, color='#CCCCCC')
    plt.plot(X0[:10], X1[:10], 'o-', ms=4, lw=2, color='blue')
    min_x0 = min(min(X0), TrueMin[0])
    max_x0 = max(max(X0), TrueMin[0])
    min_x1 = min(min(X1), TrueMin[1])
    max_x1 = max(max(X1), TrueMin[1])
    if lim == 0:
        xlim_min, xlim_max = min_x0-0.1*abs(min_x0-max_x0), max_x0+0.1*abs(min_x0-max_x0)
        ylim_min, ylim_max = min_x1-0.1*abs(min_x1-max_x1), max_x1+0.1*abs(min_x1-max_x1)
    else:
        xlim_min, xlim_max, ylim_min, ylim_max = -lim, lim, -lim, lim
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    x_points = np.linspace(xlim_min, xlim_max, 500)
    y_points = np.linspace(ylim_min, ylim_max, 500)
    Xgrid, Ygrid = np.meshgrid(x_points, y_points)
    Z = np.zeros(Xgrid.shape)
    for i in range(Xgrid.shape[0]):
        for j in range(Xgrid.shape[1]):
            Z[i][j] = f(np.array([Xgrid[i][j], Ygrid[i][j]]))
    plt.contour(Xgrid, Ygrid, Z)
    plt.xlabel('$x_0$', fontsize=24)
    plt.ylabel('$x_1$', fontsize=24)
    plt.savefig(OutName+'.png')
    # plt.show()
    plt.close()
