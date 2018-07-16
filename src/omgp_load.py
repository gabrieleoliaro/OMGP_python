#!/usr/bin/python
from __future__ import division
import numpy as np
from covariance import *


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
    
    return [x, Y]


def omgp_load(x, Y, cluster_indexes, window_indexes, clust_min=0, clust_max=np.inf, window_min=0, window_max=np.inf, minSamplesxWindow=0, maxSamplesxWindow=np.inf):
    if window_min < 0 or window_min >= len(window_indexes):
        window_min = 0
    if window_max < 0 or window_max >= len(window_indexes):
        window_max = len(window_indexes) - 1
    if clust_min < 0 or clust_min >= len(cluster_indexes):
        clust_min = 0
    if clust_max < 0 or clust_max >= len(cluster_indexes):
        clust_max = len(cluster_indexes) - 1
    if minSamplesxWindow < 0 or minSamplesxWindow > 9:
        minSamplesxWindow = 0
    if maxSamplesxWindow < 0 or maxSamplesxWindow > 9:
        maxSamplesxWindow = 9

    x_input = np.array([], dtype=int)
    y_input = np.array([])
    for i in range(len(cluster_indexes)):
        if i < clust_min or i > clust_max:
            continue
        for w in range(window_min, window_max):
            sample_min = window_indexes[w]
            sample_max = window_indexes[window_max] if w == window_max-1 else window_indexes[w+1]
            if i == len(cluster_indexes) - 1:
                # Go from here to end of x/Y array
                x_plot = np.array([], dtype=int)
                Y_plot = np.array([])
                for j in range(cluster_indexes[i], len(x)):
                    if x[j] >= sample_min and x[j] < sample_max:
                        x_plot = np.append(x_plot, x[j])
                        Y_plot = np.append(Y_plot, Y[j])
                if len(x_plot) >= minSamplesxWindow and len(x_plot) <= maxSamplesxWindow:
                    x_input = np.append(x_input, x_plot)
                    y_input = np.append(y_input, Y_plot)
            else:
                # Go from here to cluster_indexes[i+1]
                x_plot = np.array([], dtype=int)
                Y_plot = np.array([])
                for j in range(cluster_indexes[i], cluster_indexes[i+1]):
                    if x[j] >= sample_min and x[j] < sample_max:
                        x_plot = np.append(x_plot, x[j])
                        Y_plot = np.append(Y_plot, Y[j])
                if len(x_plot) >= minSamplesxWindow and len(x_plot) <= maxSamplesxWindow:
                    x_input = np.append(x_input, x_plot)
                    y_input = np.append(y_input, Y_plot)

    x_input = np.matrix(x_input).conj().transpose()
    y_input = np.matrix(y_input).conj().transpose()


    return [x_input, y_input]