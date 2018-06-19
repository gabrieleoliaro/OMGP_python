#!/usr/bin/python

import numpy as np

def omgp_gen(loghyper, n, D, m):
    """
    [x, Y] = omgp_gen(loghyper, n, D, m)

    Generate n output data points for m GPs. Each data point is D
    dimensional. The inputs are unidimensional.

    loghyper collects the process hyperparameters [log(timescale); 0.5*log(signalPower); 0.5*log(noisePower)]
    """
    covfunc = np.array(['covSum', 'covSEiso', 'covNoise']) # TODO: specify which covSum, covSEiso, covNoise function

    x = np.zeros((n * m, 1))
    Y = np.zeros((n * m, D))
    for k in range(m):
        x[(k - 1) * n + 1 : k * n) = np.random.random((n, 1)) * (n - 1) + 1
        Y[(k-1) * n + 1 : k * n, :] = np.matmul(np.linalg.cholesky(covfunc[0](loghyper, x[(k-1) * n + 1 : k * n])), np.random.standard_normal((n, D)))        # Cholesky decomp.

    order_X = np.argsort(x)
    x = np.sort(x)
    Y = np.sort(Y, order=order_X)
    
    
