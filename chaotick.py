import math
import numpy as np
import matplotlib.pyplot as plt

class FailureToConvergeException(Exception):
    pass

class NumericalFailureException(Exception):
    pass

# Press, 2007, p362
# Newton-Raphson Method for finding 1D roots
def newton_raphson(f_x, fprime_x, x_min, x_max, x_acc=0.0001, n=100):
    guess = 0.5 * (x_max + x_min)
    for i in range(n):
        f = f_x(guess)
        df = fprime_x(guess)
        dx = f/df
        guess -= dx
        if guess < x_min or guess > x_max:
            raise NumericalFailureException("Jumped out of bounds: %f" % guess)
        if dx < x_acc:
            return guess
    raise FailureToConvergeException("Failed to converge, best guess=%f" % (guess))

def logistic_equation(X, r=1):
    return r*X*(1 - X) 

# Strogatz, Nonlinear Dynamics and Chaos, pg 33-34
# 4th order Runge Kutta numerical approximation
def runge_kutta_4(dydx, range=(0,10), step=0.1, init_val=0.0):

    cur_val = init_val    
    res = []
    for i in np.arange(range[0], range[1], step):
        res.append(cur_val)
        k1 = dydx(cur_val) * step
        k2 = dydx(cur_val + 0.5*k1) * step
        k3 = dydx(cur_val + 0.5*k2) * step
        k4 = dydx(cur_val + k3) * step
        cur_val += (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    return res

# code below adapted from https://medium.com/@olutosinbanjo/how-to-plot-a-direction-field-with-python-1fd022e2d8f8

def gen_meshgrid(x_interval=[-3,3], y_interval=[-3,3], n=20):
    x = np.arange(x_interval[0], x_interval[1], (x_interval[1] - x_interval[0]) / float(n))
    y = np.arange(y_interval[0], y_interval[1], (y_interval[1] - y_interval[0]) / float(n))
    return np.meshgrid(x, y)

def gen_slopevals(X, Y, dydx):
    dy = dydx(Y)
    dx = np.ones(dy.shape)
    return (dy, dx)

def plot_slope_field_prepped(x_ticks, y_ticks, x_vals, y_vals, normalized=True, title="Slope Field", curves=[], ax=None):

    if normalized:
        x_vals = x_vals / np.sqrt(x_vals**2 + y_vals**2)
        y_vals = y_vals / np.sqrt(x_vals**2 + y_vals**2)

    if ax is None:
        plot = plt.figure()
        ax = plot.gca()

    ax.quiver(x_ticks, y_ticks, x_vals, y_vals, 
               headlength=7,
               color='Teal')
    for x,y in curves:
        ax.plot(x, y)

    ax.set_title(title)
    #plt.show(plot)

def plot_slope_field(x_interval, y_interval, dydx, ticks=None, normalized=True, title="Slope Field", curves=[], ax=None):
    X, Y = gen_meshgrid(x_interval, y_interval, ticks)
    dy, dx = gen_slopevals(X, Y, dydx)
    return plot_slope_field_prepped(X, Y, dx, dy, ax=ax, normalized=normalized, title=title, curves=curves)

# code below taken from ipython cookbook
# https://ipython-books.github.io/121-plotting-the-bifurcation-diagram-of-a-chaotic-dynamical-system/
def plot_system(r, x0, n, ax=None):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(0, 1)
    ax.plot(t, logistic_equation(t, r=r), 'k', lw=2)
    ax.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = logistic_equation(x, r)
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'k', lw=1)
        ax.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=10,
                alpha=(i + 1) / n)
        x = y

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")

if __name__ == '__main__':
    for i in range(15):
        try:
            print("%f: %f" % (i, newton_raphson(math.sin, math.cos, 3.0, 3.5, n=i, x_acc=0.0000001)))
        except FailureToConvergeException as e:
            print("%f: %s" % (i, e))
