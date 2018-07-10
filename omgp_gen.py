#!/usr/bin/python

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
    Y = np.zeros((n * m, 1))
    
    if covfunc[0] == 'covSum':
        function0 = covSumCM
    elif covfunc[0] == 'covSEiso':
        function0 = covSEisoCM
    elif covfunc[0] == 'covNoise':
        function0 = covNoiseCM

    
    for k in range(m):
        x[k * n : (k + 1) * n] = np.random.rand(n, 1) * (n - 1) + 1
        Y[k * n : (k + 1) * n] = np.matmul(np.linalg.cholesky(function0(covfunc, loghyper, x[k * n : (k + 1) * n])), np.random.randn(n,1))       # Cholesky decomp. np.random.standard_normal((n, D))

    # Make sure x is a column vector, and not a row vector
    if (x.shape[1] is not 1):
        x = x.conj().transpose()

    order_X = ([i[0] for i in sorted(enumerate(x), key=lambda x:x[1])])
    x.sort(0)
    
    Y = Y[order_X, :]
    
    return [x, Y]
