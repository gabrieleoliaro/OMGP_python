#!/usr/bin/python
from __future__ import division
import numpy as np
from covariance import *
from parser import *


def omgp_gen(loghyper, n, D, m):
    """
    [x, Y] = omgp_gen(loghyper, n, D, m)

    Generate n output data points for m GPs. Each data point is D
    dimensional. The inputs are unidimensional.

    loghyper collects the process hyperparameters [log(timescale); 0.5*log(signalPower); 0.5*log(noisePower)]
    """
    # Specify which functions to use to compute the covariance matrix
    covfunc = np.array(['covSum', 'covSEiso', 'covNoise'])

    
    x = np.matrix((np.ones((n * m, 1))))
    Y = np.zeros((n * m, 1)) if D == 2 else np.zeros((n * m, D))
    
    if covfunc[0] == 'covSum':
        function0 = covSumCM
    elif covfunc[0] == 'covSEiso':
        function0 = covSEisoCM
    elif covfunc[0] == 'covNoise':
        function0 = covNoiseCM

    
    for k in range(m):
        x[k * n : (k + 1) * n] = np.random.rand(n, 1) * (n - 1) + 1
        if D == 2:
            Y[k * n : (k + 1) * n] = np.matmul(np.linalg.cholesky(function0(covfunc, loghyper, x[k * n : (k + 1) * n])), np.random.randn(n,1))       # Cholesky decomp. np.random.standard_normal((n, D))
        else:
            Y[k * n : (k + 1) * n, :] = np.matmul(np.linalg.cholesky(function0(covfunc, loghyper, x[k * n : (k + 1) * n])), np.random.randn(n,D))       # Cholesky decomp. np.random.standard_normal((n, D))

            
    # Make sure x is a column vector, and not a row vector
    if (x.shape[1] is not 1):
        x = x.conj().transpose()

    order_X = ([i[0] for i in sorted(enumerate(x), key=lambda x:x[1])])
    x.sort(0)
    
    Y = Y[order_X, :]

    x_train = x[::2]
    Y_train = Y[::2]
    x_test = x[1::2]
    Y_test = Y[1::2]
    
    return [x_train, Y_train, x_test, Y_test]


def omgp_load(filename):
    windows = new_parse(filename)

    # plotting some windows for inspection
    print_windows(windows, start_from=0)

    # get windows with at least 9 points
    Y = []
    for w in windows:
        for c in w.values():
            if len(c) >= 9:
                Y.append(c)
    # ---

    # plotting lines with at least 9 points
    fig = plt.figure()
    for y in Y:
        plt.plot(y)
    plt.title("all lines with at least 9 samples")  # Add title
    plt.xlabel('sample')
    plt.ylabel('angle')
    plt.show()
    # ---

    # setting the data set properly
    x_train = np.array([], dtype=int)
    for y in Y:
        x_train = np.append(x_train, range(len(y)))
    x_train = np.matrix(x_train).conj().transpose()
    Y_train = np.array([item for sublist in Y for item in sublist])
    Y_train = np.matrix(Y_train).conj().transpose()
    x_test = np.random.randint(0,9, (9,1))
    x_test.sort(0)

    return [x_train, Y_train, x_test]

def print_windows(windows, start_from=0, n_win=16, min_pts=9):
    """Plot a sequence of windows
    Parameters:
        windows: the list of windows
        start_from: index of first window to be plotted.
        n_win: how many windows to plot
        min_pts: the minimum number of points to be plotted.
    """
    fig = plt.figure()
    for i, w in enumerate(range(start_from, start_from + n_win)):
        l = int(np.sqrt(n_win))
        ax = fig.add_subplot(l, l, i + 1)
        for c in windows[w].values():
            if len(c) >= min_pts:
                ax.plot(c)
        ax.set_ylim((89, 181))
        ax.set_xlabel('sample')
        ax.set_ylabel('angle')
        ax.set_yticks(range(90, 180, 20))
        ax.set_xticks(range(0, 10, 3))
        ax.grid()

    plt.tight_layout()
    plt.show()
