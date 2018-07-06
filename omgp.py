#!/usr/bin/python

import numpy as np
import numpy.matlib as npm
import warnings

from test import *
# External modules
#from omgpbound import *
from minimize import *
from covSEiso import *
from covNoise import *
from omgpEinc import *


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

 
    Y = np.divide(Y - np.matmul(np.ones((N, 1)), np.matrix((meany))), np.matmul(np.ones((N, 1)), np.matrix((stdy))) + 1e-6)

    # Independent or shared hyperparameters
    if len(covfunc) == M:
        print('Using a different set of hyperparameters for each component')
    else:
        print('Using shared hyperparameters for all components')

    # Initial hyperparameters setting
    lengthscale = np.log(np.mean((X.max() - X.min()).conj().transpose() / 2/5))
    lengthscale = np.maximum(lengthscale, -100) # minimum value of lengthscale is -100
    covpower = 0.5* np.log(1)
    noisepower = 0.5 * np.log(1/8) * np.ones((M, 1))

    loghyper = np.array(())
    if covfunc.shape != (1,1):
        covfunc_array = np.squeeze(np.asarray(covfunc))
    else:
        covfunc_array = covfunc
    for function in covfunc_array:
        if function == 'covNoise':
            loghyper = loghyper + covpower
        elif function == 'covSEiso':
            loghyper = np.append(loghyper, lengthscale)
            loghyper = np.append(loghyper, covpower)
        else:
            raise Warning('Covariance type not (yet) supported')
    # Add responsibilities
#    qZ = np.random.rand(N, M) + 10
    qZ = read_matrix('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/qZ')

#    qZ = np.divide(qZ, npm.repmat(qZ.sum(axis = 1),M,1).conj().transpose())
    qZ = np.divide(qZ, npm.repmat(qZ.sum(axis = 1),1,M))
    logqZ = np.log(qZ)
    
    logqZ = logqZ - np.matmul(np.matrix((logqZ [:,0])), np.ones((1, M))) #np.matrix(()).conj().transpose()
    logqZ = logqZ[:,1:]


    loghyper = np.append(loghyper, np.zeros((M - 1, 1)))
    loghyper = np.append(loghyper, noisepower)
    loghyper = np.append(loghyper, logqZ.flatten('F').conj().transpose())

    # Iterate EM updates
    F_old = np.inf


    convergence = []
    for iter_variable in range(maxiter):
        [loghyper, conv1] = omgpEinc(loghyper, covfunc, M, X, Y)
        print('\nBound after E-step is %.4f' % (conv1[-1]))
        [loghyper, conv2] = minimize(loghyper, 'omgpbound', 10, 'learnhyp', covfunc, M, X, Y)
 
        convergence = np.concatenate((conv1, conv2))
        F = convergence[-1]
        if np.abs(F - F_old) < np.abs(F_old) * (1e-5):
            break
    
        F_old = F
    if iter_variable is maxiter:
        print('Maximum number of iterations exceeded')
    print('\n')


    # Final pass, also updating pi0
    [loghyper, conv] = minimize(loghyper, 'omgpbound', 20, 'learnall', covfunc, M, X, Y)

    print('\n\n')
    
    F = conv[-1]
    
    loghyperinit = np.concatenate((loghyper[ : -N * (M - 1) - 2 * M + 1], loghyper[-N * (M - 1) - M : -N * (M - 1)]))

    # Compute qZ and pi0
    logqZ = np.concatenate((np.zeros((N, 1)), np.reshape(loghyper[-N * (M - 1) :], (N, M - 1), order='F')), axis=1)
    qZ = np.exp(logqZ - np.matrix((logqZ.max(axis=1))).conj().transpose() * np.ones((1, M)))
    qZ = np.divide(qZ, (qZ.sum(axis=1) * np.ones((1, M))))
    
    logpZ = np.concatenate(([0], loghyper[-N * (M - 1) -2 * M +1 : -N * (M - 1) - M]))
    logpZ = logpZ - logpZ.max()
    logpZ = logpZ - np.log(np.exp(logpZ).sum())
    
    pi0 = np.exp(logpZ)

    Ntst = Xs.shape[0]
    [mu, C] = omgpboundB(loghyper, 'learnall', covfunc, M, X, Y, Xs)

    mu1 = np.ones((Ntst, 2, M))
    mu2 = np.ones((Ntst, 2, M))
    C1 = np.ones((Ntst, 2, M))
    for i in range(M):
        mu1[:,:,i] = np.multiply(mu1[:,:,i], np.kron(np.ones((30,1)), meany))
        mu2[:,:,i] = np.multiply(mu2[:,:,i], np.kron(np.ones((30,1)), (stdy + 1e-6)))
        C1[:,:,i] = np.multiply(C1[:,:,i], np.kron(np.ones((30,1)), np.power((stdy + 1e-6), 2)))
    
    mu = mu1 + np.multiply(mu, mu2)
    C = np.multiply(C, C1)
    
    return [F, qZ, loghyperinit, mu, C, pi0]
