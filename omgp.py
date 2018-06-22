#!/usr/bin/python

import numpy as np
import numpy.matlib as npm
import warnings

from test import *
# External modules
#from omgpEinc import *
#from omgpbound import *
#from minimize import *
from covSEiso import *
from covNoise import *


#omgpA -> nargin = 5, nargout = 6, called first from test:omgp

def omgpA(covfunc, M, X, Y, Xs):
    """
    One possible way to initialize and optimize hyperparameters:

    Uses omgpEinc to perform the update of qZ and omgpbound to update
    hyperparameters.
    """
    if (np.argsort(X).sum() != 0) or X.shape[1] != 1:
        warnings.warn('Your *inputs* are multidimensional or not sorted. Optimization may have trouble converging! (Multidimensional outputs are OK).') 

    # Initialization
    [N, oD] = Y.shape
    maxiter = 100
    
    #Assume zero-mean functions, substract mean 
    meany = np.mean(Y, axis=0)
    stdy = np.std(Y, axis=0, ddof=1) # Setting the ddof (degrees of freedom) param  to 1 to get the same value as Matlab's
    Y = np.divide(Y - np.matmul(np.ones((N, 1)),  meany), np.matmul(np.ones((N, 1)), stdy) + (10 ** -6))

    # Independent or shared hyperparameters
    if len(covfunc) == M:
        print('Using a different set of hyperparameters for each component')
    else:
        print('Using shared hyperparameters for all components')

    # Initial hyperparameters setting
    lengthscale = np.log(np.mean((X.max() - X.min()).conj().transpose() / 2/5))
    lengthscale = np.maximum(lengthscale, -100) # minimum value of lengthscale is -100
    covpower = 0.5 * np.log(1)
    noisepower = 0.5 * np.log(1/8) * np.ones((M, 1))

    loghyper = np.array(())
    for function in covfunc:
        if function == covNoiseCM or function == covNoiseTSC or function == covNoiseDERIV:
            loghyper = loghyper + covpower
        elif function == covSEisoCM or function == covSEisoTSC or function == covSEisoDERIV:
            loghyper = np.append(loghyper, lengthscale)
            loghyper = np.append(loghyper, covpower)
        else:
            raise Warning('Covariance type not supported')

    # Add responsibilities
    qZ = np.random.rand(N, M) + 10
    print_matrix(npm.repmat(qZ.sum(axis = 1),M,1).conj().transpose())
    qZ = np.divide(qZ, npm.repmat(qZ.sum(axis = 1),M,1).conj().transpose())
    logqZ = np.log(qZ)
    logqZ = logqZ - np.matmul(np.matrix((logqZ [:,0])).conj().transpose(), np.ones((1, M)))
    logqZ = logqZ[:,1:]
    print_matrix(np.zeros((M - 1, 1)), 'zeros')
    print_matrix(noisepower, 'noisepower')
    print_matrix(logqZ[:],'logqZ')
    print(np.zeros((M - 1, 1)).shape)
    print(noisepower.shape)
    print(logqZ[:].shape)
    loghyper = loghyper + np.zeros((M - 1, 1)) + noisepower +logqZ[:]

    # Iterate EM updates
    F_old = inf
    convergence = []
    for iter_variable in range(maxiter):
        [loghyper, conv1] = omgpEinc(loghyper, covfunc, M, X, Y)
        print('Bound after E-step is %.4f\n' % (conv1[-1]))
        [loghyper, conv2] = minimize(loghyper, 'omgpbound', 10, 'learnhyp', covfunc, M, X, Y)
        convergence = np.concatenate((conv1, conv2))
        F = convergence[-1]
        if np.abs(F - F_old) < np.abs(F_old) * (10 ** -6):
            break
    
        F_old = F
    if iter_variable is maxiter:
        print('Maximum number of iterations exceeded')

    # Final pass, also updating pi0
    [loghyper, conv] = minimize(loghyper, 'omgpbound', 20, 'learnall', covfunc, M, X, Y)
    F = conv[-1]
##    loghyperinit = [loghyper[1 : -N * (M - 1) - 2 * M + 1]
##    loghyper[-N * (M - 1) - M : -N * (M - 1)]
##
##    # Compute qZ and pi0
##    logqZ = [zeros(N,1) reshape(loghyper(end-N*(M-1)+1:end),N,M-1)]
##    qZ = np.exp(logqZ - max(logqZ,[],2)*ones(1,M))
##    qZ = np.divide(qZ, (qZ.sum(axis=1) * np.ones((1, M))))
##
##    logpZ = [0; loghyper[-N * (M - 1) -2 * M +1 : -N * (M - 1) - M]].conj().transpose() 
##    logpZ = logpZ - np.maximum(logpZ)
##    logpZ = logpZ - np.log(np.exp(logpZ).sum())
##    pi0 = np.exp(logpZ)
##
##    Ntst = Xs.shape[0]           
##    [mu, C] = omgpbound(loghyper, 'learnall', covfunc, M, X, Y, Xs)
##    mu = repmat(meany, [Ntst, 1, M]) + np.multiply(mu,repmat(stdy + 1e-6, [Ntst, 1, M]))
##    C = np.multiply(C, repmat((stdy + 1e-6) ** 2, [Ntst, 1, M]))
##
##    
##    return [F, qZ, loghyperinit, mu, C, pi0]
