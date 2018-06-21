#!/usr/bin/python

import numpy as np
from covNoise import *
from covSEiso import *
from covSum import *
from test import print_matrix

def omgp_gen(loghyper, n, D, m):
    """
    [x, Y] = omgp_gen(loghyper, n, D, m)

    Generate n output data points for m GPs. Each data point is D
    dimensional. The inputs are unidimensional.

    loghyper collects the process hyperparameters [log(timescale); 0.5*log(signalPower); 0.5*log(noisePower)]
    """
    # Specify which functions to use to compute the covariance matrix
    covfunc = np.array([covSumCM, covSEisoCM, covNoiseCM])
    
    #x = np.zeros((n * m, 1))
    x = np.matrix([1.59179383164442, 1.98724574822052, 2.36805747806268, 2.57961442242197, 2.90841815579740, 2.98582821189663, 3.36554886433977, 3.69193099174221, 3.87333272293030, 4.06586989646427, 4.24775347062622, 4.25822887122662, 4.60562669927342, 5.90147028151262, 6.40973058143213, 6.49872619872079, 6.65290582882933, 7.18852312216145, 7.27574578712792, 7.30643593854807, 7.51216525729664, 8.51594637738672, 9.26331116129353, 9.49114157234113, 9.68550806384046, 9.91614552176422, 10.0995600566108, 10.1490497082835, 10.6411430600643, 10.8273562367179, 11.3435116980159, 11.4002695752538, 11.6662083365186, 11.7000741044571, 11.8616109514322, 11.9105625713640, 12.1198018095748, 13.4310862306541, 13.5007381766079, 13.8568878541673, 13.8998606835519, 13.9607356118026, 14.0162735719598, 14.2863840897013, 15.2960387791029, 15.3848391099917, 15.7396563153646, 15.7516826384958, 15.8961175338559, 15.9057658656030, 16.0940203686051, 16.1402836886706, 17.4375900900850, 17.4760238527489, 17.6193427650044, 17.7599776163929, 18.6337951572861, 18.9465819437975, 19.1599337434421, 19.5751526079905])
    x = x.conj().transpose()
    Y = np.zeros((n * m, D))
    for k in range(m):
        #x[k * n : (k + 1) * n] = np.random.random((n, 1)) * (n - 1) + 1
        Y[k * n : (k + 1) * n, :] = np.matmul(np.linalg.cholesky(covfunc[0](covfunc, loghyper, x[k * n : (k + 1) * n])), np.ones((n,D)))        # Cholesky decomp. np.random.standard_normal((n, D))


        
    order_X = np.argsort(x)
    print_matrix(order_X, 'ordrx')
    length_X = 0
    
    if (x.shape[0]) is not 1:
        length_X = x.shape[0]
    else:
        length_X = x.shape[1]
        
    x = np.sort(x)
    print_matrix(x, 'x')
    # fix the argsort problem
    y_sorted = Y
    print_matrix(y_sorted)
    
    for i in range(length_X):
        y_sorted[i] = Y[order_X[i]]
    print_matrix(Y, 'Y')
    
    return [x, Y]
