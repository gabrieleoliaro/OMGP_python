#!/usr/bin/python

import numpy as np
from covNoise import *
from covSEiso import *
from covSum import *

def omgp_gen(loghyper, n, D, m):
    """
    [x, Y] = omgp_gen(loghyper, n, D, m)

    Generate n output data points for m GPs. Each data point is D
    dimensional. The inputs are unidimensional.

    loghyper collects the process hyperparameters [log(timescale); 0.5*log(signalPower); 0.5*log(noisePower)]
    """
    # Specify which functions to use to compute the covariance matrix
    covfunc = np.array(['covSumCM', 'covSEisoCM', 'covNoiseCM'])
    
    x = np.zeros((n * m, 1))
    Y = np.zeros((n * m, D))
    for k in range(m):
        x[k * n : (k + 1) * n] = np.random.random((n, 1)) * (n - 1) + 1
        print((Y[k * n : (k + 1) * n, :]).shape)
        Y[k * n : (k + 1) * n, :] = np.matmul(np.linalg.cholesky(covfunc[0](loghyper, x[k * n : (k + 1) * n])), np.random.standard_normal((n, D)))        # Cholesky decomp.


    order_X = np.argsort(x)
    x = np.sort(x)
    Y = np.sort(Y, order=order_X)
    
    return [x, Y]
