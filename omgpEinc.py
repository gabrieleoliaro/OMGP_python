#!/usr/bin/python
import numpy as np
from covNoise import *
from covSEiso import *
from test import *

def omgpEinc(loghyper, covfunc, M, X, Y):
    """
    Update elements 1:n in qZ at iteration n, then 100 iterations more
    updating all the elements in qZ
    """
    
    #--- Initialize
    #[N, D] = X.shape
    [N, oD] = Y.shape

    maxit = N + 100
    logqZ = np.concatenate((np.zeros((N, 1)), np.reshape(loghyper[-N * (M - 1) : ], (M-1, N)).conj().transpose()), axis=1)
    logqZ = logqZ - np.matmul(np.matrix((np.max(logqZ, axis = 1))).conj().transpose(), np.ones((1, M)))
    logqZ = logqZ - np.matmul(np.log(np.exp(logqZ).sum(axis=1)), np.ones((1, M)))
    qZ = np.exp(logqZ)
    sn2 = np.matmul(np.ones((N, 1)), np.matrix((np.exp(2 * loghyper[-N * (M - 1) - M : -N * (M - 1)]))))
    logpZ = np.concatenate(([0], np.array(loghyper[-N * (M - 1) -2 * M + 1 : -N * (M - 1) - M])), axis=0).conj().transpose()
    logpZ = logpZ - np.max(logpZ)
    logpZ = logpZ - np.log(np.exp(logpZ).sum(axis=0))
    logpZ = np.ones((N, 1)) * logpZ
    
    convergence = np.zeros((N * maxit, 1))

    oldFant = np.inf
 
    for iter_variable in range(maxit):
        sqB = np.sqrt(np.divide(qZ, sn2))
    
        # --- Contribution of independent (modified) GPs
        oldF = 0
        hypstart = 0
        a = np.zeros((N, M))

        covfunc_array = np.asarray(covfunc)
        cm = covfunc_array.item(0)

        if cm == 'covNoise':
            numhyp = 1
            K = covNoiseCM(loghyper[0 : numhyp], X)
        elif cm == 'covSEiso':
            numhyp = 2
            K = covSEisoCM(loghyper[0 : numhyp], X)
        else:
            raise Warning('Covariance type not (yet) supported')
        
        for m in range(M):
            if len(covfunc_array) > 1:
                cm = covfunc_array.item(m)
                if cm == 'covNoise':
                    numhyp = 1
                    K = covNoiseCM(loghyper[hypstart : hypstart + numhyp], X)
                elif cm == 'covSEiso':
                    numhyp = 2
                    K = covSEisoCM(loghyper[hypstart : hypstart + numhyp], X)
                else:
                    raise Warning('Covariance type not (yet) supported')
            else:
                hypstart = 1

            R = (np.linalg.cholesky(np.eye(N) + np.multiply(K, np.matmul(sqB[:, m], sqB[:, m].conj().transpose())))).conj().transpose()
            sqBY = np.multiply(np.matmul(sqB[:, m], np.ones((1, oD))), Y)
            v = np.linalg.solve(R.conj().transpose(), sqBY)
            if not np.allclose(np.dot(R.conj().transpose(), v), sqBY):
                raise Warning(" linalg.solve not successful")
            oldF = oldF + 0.5 * np.power(v, 2).sum() + oD * np.log(np.diag(R)).sum(axis=0)
            diag_sqB = np.zeros((N, N))
            np.fill_diagonal(diag_sqB, sqB[:, m])
            U = np.linalg.solve(R.conj().transpose(), diag_sqB)
            if not np.allclose(np.dot(R.conj().transpose(), U), diag_sqB):
                raise Warning(" linalg.solve not successful")
            alpha = np.matmul(U.conj().transpose(), v)

            diagSigma = (np.diag(K) - np.power(np.matmul(U, K), 2).sum(axis=0).conj().transpose())[:, 0]
            mu = np.matmul(K, alpha)
            #a = a.conj().transpose()
            #a[m] = np.multiply(np.divide(-0.5, sn2[:,m]), (np.power((Y - mu), 2).sum(axis=1) + np.multiply(oD, diagSigma))).conj().transpose()
            #a = a.conj().transpose()
            a[:, m] = (np.multiply(np.divide(-0.5, sn2[:,m]), (np.power((Y - mu), 2).sum(axis=1) + np.multiply(oD, diagSigma))).conj().transpose()).flatten('F')
            
            hypstart = hypstart + numhyp

        if (hypstart + 2 * M + N * (M - 1) - 2) != len(loghyper):
            raise Warning('Incorrect number of parameters')
            
        KLZ = np.multiply(qZ, (logqZ - logpZ)).sum() # KL Divergence from the posterior to the prior on Z
        oldF = oldF + oD / 2 * (np.multiply(qZ, np.log(2 * np.pi * sn2))).sum() + KLZ
        convergence[iter_variable] = oldF

        temp = a + logpZ - oD / 2 * np.log(2 * np.pi * sn2)
        logqZ[0 : min(iter_variable + M, N)+1, :] = temp[0 : min(iter_variable + M, N)+1, :]

        logqZ = logqZ - np.multiply(np.max(logqZ, axis=1), np.ones((1, M)))
        logqZ = logqZ - np.multiply(np.log(np.exp(logqZ).sum(axis=1)), np.ones((1, M)))

        qZ = np.exp(logqZ)
    
        if (iter_variable + M > N) and (abs(oldF - oldFant) < abs(oldFant) * (1e-6)):
            break
        
        oldFant = oldF
    convergence = convergence[0 : iter_variable+1]
    logqZ = logqZ - logqZ[:, 0] * np.ones((1, M))
    logqZ = logqZ[:, 1:]
    loghyper[-N * (M - 1) : ] = logqZ.flatten('F')

    # This would also update pZ:
    #logpZ = log(sum(qZ,1)+ones(1,M)/M);
    #logpZ = logpZ - max(logpZ);logpZ = logpZ-log(sum(exp(logpZ)));logpZ=logpZ-logpZ(1); 
    #loghyper(end-N*(M-1)-2*M+2:end-N*(M-1)-M) = logpZ(2:end);

    return [loghyper, convergence]
