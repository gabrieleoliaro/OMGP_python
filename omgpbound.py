#!/usr/bin/python

import numpy as np

# TODO: replace A, B, C, ... with number of params required
# A -> nargin==6
# A2 -> nargin==6 && nargout==2
# B -> nargin == 7
# B2 -> nargin == 7 && nargout ==2

def omgpboundA(loghyper, learn, covfunc, M, X, Y, Xs):
    """
    Computes the negative of the Marginalized Variational Bound (F) and its 
    derivatives wrt loghyper (dF).

    Parameters:
    loghyper: K hyperparameters, pZ, sn2 (M trajectories), logqZ
    learn: 'learnqZ', 'learnhyp', 'learnall'
    covfunc: Array of covariance functions. If it is a single one, it is shared.
    M: Number of trajectories
    X, Y, Xs: Inputs, outputs, test inputs

    (c) Miguel Lazaro-Gredilla 2010
    """

    # Initialize
    # [N, D] = X.shape
    [N, oD] = Y.shape

    logqZ = np.concatenate((np.zeros((N, 1)), loghyper[-N*(M-1):].reshape(N, M-1, order='F')))
    logqZ = logqZ - logqZ.max(1) * np.ones((1, M))
    logqZ = logqZ - np.log((np.exp(logqZ)).sum(axis=1)) * np.ones((1, M))
    qZ = np.exp(logqZ)
    sn2 = np.ones((N, 1)) * np.exp(2 * loghyper[-N*(M-1)-M:-N*(M-1)]).conj().transpose()
    logpZ = np.concatenate((np.array([0]), loghyper[-N*(M-1)-2*M+1: -N*(M-1)-M])).conj().transpose()
    logpZ = logpZ - logpZ.max(0)
    logpZ = logpZ - np.log(np.exp(logpZ).sum(axis = 0))
    logpZ = np.ones((N, 1)) * logpZ
    sqB = np.sqrt(np.divide(qZ, sn2))
    dlogqZ = np.zeros((N, M))

    # Contribution of independent (modified) GPs
    F = 0
    dF = np.zeros((loghyper.shape))

    hypstart = 1
    cm = covfunc[0]
    numhyp = # TODO: numhyp = eval(feval(cm{:})); K = feval(cm{:}, loghyper(1:numhyp), X);
    for m in M:
        if len(covfunc) is > 1:
            #TODO: cm = covfunc{m};numhyp = eval(feval(cm{:}));
            # TODO: K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
        else:
            hypstart = 1

        R = np.linalg.cholesky(np.identity(N) + K * (np.matmul(sqB[:,m], sqB[:,m].conj().transpose())).conj().transpose())
        sqBY = np.matmul(sqB[:,m], np.ones((1,oD))) * Y
        v = np.linalg.solve(R.conj().transpose(), sqBY)
        F = F + 0.5 * (v**2).sum() + oD* np.log(np.diag(R)).sum(axis=0)
        
        hypstart = hypstart + numhyp

    if hypstart + 2 * M + N * (M - 1) - 2 is not len(loghyper):
        raise RuntimeError('Incorrect number of parameters')

    
    KLZ = (qZ * (logqZ-logpZ)).sum()                    # KL Divergence from the posterior to the prior on Z
    F = F + np.divide(oD, 2 * (qZ * np.log(2 * np.pi * sn2)).sum()) + KLZ

def omgpboundA2(loghyper, learn, covfunc, M, X, Y, Xs):
    """
    Computes the negative of the Marginalized Variational Bound (F) and its 
    derivatives wrt loghyper (dF).

    Parameters:
    loghyper: K hyperparameters, pZ, sn2 (M trajectories), logqZ
    learn: 'learnqZ', 'learnhyp', 'learnall'
    covfunc: Array of covariance functions. If it is a single one, it is shared.
    M: Number of trajectories
    X, Y, Xs: Inputs, outputs, test inputs

    (c) Miguel Lazaro-Gredilla 2010
    """

    # Initialize
    # [N, D] = X.shape
    [N, oD] = Y.shape

    logqZ = np.concatenate((np.zeros((N, 1)), loghyper[-N*(M-1):].reshape(N, M-1, order='F')))
    logqZ = logqZ - logqZ.max(1) * np.ones((1, M))
    logqZ = logqZ - np.log((np.exp(logqZ)).sum(axis=1)) * np.ones((1, M))
    qZ = np.exp(logqZ)
    sn2 = np.ones((N, 1)) * np.exp(2 * loghyper[-N*(M-1)-M:-N*(M-1)]).conj().transpose()
    logpZ = np.concatenate((np.array([0]), loghyper[-N*(M-1)-2*M+1: -N*(M-1)-M])).conj().transpose()
    logpZ = logpZ - logpZ.max(0)
    logpZ = logpZ - np.log(np.exp(logpZ).sum(axis = 0))
    logpZ = np.ones((N, 1)) * logpZ
    sqB = np.sqrt(np.divide(qZ, sn2))
    dlogqZ = np.zeros((N, M))

    # Contribution of independent (modified) GPs
    F = 0
    dF = np.zeros((loghyper.shape))

    hypstart = 1
    cm = covfunc[0]
    numhyp = # TODO: numhyp = eval(feval(cm{:})); K = feval(cm{:}, loghyper(1:numhyp), X);
    for m in M:
        if len(covfunc) is > 1:
            #TODO: cm = covfunc{m};numhyp = eval(feval(cm{:}));
            # TODO: K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
        else:
            hypstart = 1

        R = np.linalg.cholesky(np.identity(N) + K * (np.matmul(sqB[:,m], sqB[:,m].conj().transpose())).conj().transpose())
        sqBY = np.matmul(sqB[:,m], np.ones((1,oD))) * Y
        v = np.linalg.solve(R.conj().transpose(), sqBY)

        F = F + 0.5 * (v**2).sum() + oD* np.log(np.diag(R)).sum(axis=0)

        # Compute derivatives
        U = np.divide(R.conj().transpose(), np.diag(sqB[:,m]))
        alpha = np.matmul(U.conj().transpose(), v)
        diagW = ((Y - K*alpha) ** 2).sum(axis=1) + np.matmul(oD, (np.diag(K) - (np.matmul(U,K) ** 2).sum(axis=0).conj().transpose()))
        
        if learn is not 'learnqZ':
            W = np.matmul(oD, (np.matmul(U.conj().transpose(), U))) - np.matmul(alpha, alpha.conj().transpose())               # Precompute for convenience
            for i = in range(numhyp):
                dF[i + hypstart - 1] = dF[i + hypstart - 1] + (W * cm(loghyper[hypstart : hypstart + numhyp - 1], X, i)).sum() / 2
            dF[- N * (M - 1) - M + m] = np.matmul(diagW.conj().transpose(), np.matmul(qZ[: , m], np.exp(-2 * loghyper[-N * (M - 1) - M + m])) * -2 / 2) # diagW * dB/dsn2

        if learn is not 'learnhyp':
            dlogqZ[:, m] = np.divide(diagW, sn2[:, m]) / 2
            
        hypstart = hypstart + numhyp

    if hypstart + 2 * M + N * (M - 1) - 2 is not len(loghyper):
        raise RuntimeError('Incorrect number of parameters')

    KLZ = (qZ * (logqZ-logpZ)).sum()                    # KL Divergence from the posterior to the prior on Z
    F = F + np.divide(oD, 2 * (qZ * np.log(2 * np.pi * sn2)).sum()) + KLZ

    if learn is not 'learnhyp':
            dKLZlogpz = ((-qZ + np.exp(logpZ)).sum(axis=0)).conj().transpose()
            dF[- N * (M - 1) - 2 * M + 1 : -N * (M - 1) - M] = dKLZlogpz[1:]  # Derivative wrt pZ
            dlogqZ = dlogqZ + logqZ - logpZ + oD / 2 * log(2 * np.pi * sn2) # Derivative wrt qZ
            dlogqZ = qZ * (dlogqZ - ((qZ * dlogqZ).sum(axis=1)) * np.ones((1, M))) # Derivative wrt actual hyperparam "beta" defnining qZ
            dlogqZ = dlogqZ[:, 1:]
            dF[-N * (M - 1) :] = dlogqZ[:]
        
    if learn is not 'learnqZ':
        dF[-N * (M - 1) - M : -N * (M - 1)] = dF[-N * (M - 1) -M : -N * (M - 1)) + oD / 2 * ((qZ * 2).sum(axis=0)).conj().transpose() # Derivative wrt sn2


def omgpboundB(loghyper, learn, covfunc, M, X, Y, Xs):
    """
    Computes the negative of the Marginalized Variational Bound (F) and its 
    derivatives wrt loghyper (dF).

    Parameters:
    loghyper: K hyperparameters, pZ, sn2 (M trajectories), logqZ
    learn: 'learnqZ', 'learnhyp', 'learnall'
    covfunc: Array of covariance functions. If it is a single one, it is shared.
    M: Number of trajectories
    X, Y, Xs: Inputs, outputs, test inputs

    (c) Miguel Lazaro-Gredilla 2010
    """

    # Initialize
    # [N, D] = X.shape
    [N, oD] = Y.shape

    logqZ = np.concatenate((np.zeros((N, 1)), loghyper[-N*(M-1):].reshape(N, M-1, order='F')))
    logqZ = logqZ - logqZ.max(1) * np.ones((1, M))
    logqZ = logqZ - np.log((np.exp(logqZ)).sum(axis=1)) * np.ones((1, M))
    qZ = np.exp(logqZ)
    sn2 = np.ones((N, 1)) * np.exp(2 * loghyper[-N*(M-1)-M:-N*(M-1)]).conj().transpose()
    logpZ = np.concatenate((np.array([0]), loghyper[-N*(M-1)-2*M+1: -N*(M-1)-M])).conj().transpose()
    logpZ = logpZ - logpZ.max(0)
    logpZ = logpZ - np.log(np.exp(logpZ).sum(axis = 0))
    logpZ = np.ones((N, 1)) * logpZ
    sqB = np.sqrt(np.divide(qZ, sn2))
    dlogqZ = np.zeros((N, M))

    # Contribution of independent (modified) GPs
    F = np.zeros((Xs.shape[0], oD, M))
    dF = np.zeros((Xs.shape[0], oD, M))

    hypstart = 1
    cm = covfunc[0]
    numhyp = # TODO: numhyp = eval(feval(cm{:})); K = feval(cm{:}, loghyper(1:numhyp), X);
    for m in M:
        if len(covfunc) is > 1:
            #TODO: cm = covfunc{m};numhyp = eval(feval(cm{:}));
            # TODO: K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
        else:
            hypstart = 1

        R = np.linalg.cholesky(np.identity(N) + K * (np.matmul(sqB[:,m], sqB[:,m].conj().transpose())).conj().transpose())
        sqBY = np.matmul(sqB[:,m], np.ones((1,oD))) * Y
        v = np.linalg.solve(R.conj().transpose(), sqBY)

        # Compute predictions
        U = np.linalg.solve(R.conj().transpose(), np.diag(sqB[:, m]))
        alpha = np.matmul(U.conj().transpose(), v)
        [Kss, Kfs] = cm(loghyper[hypstart : hypstart + numhyp - 1], X, Xs)
        F[:, :, m] = np.matmul(Kfs.conj().transpose(), alpha)
                    
        hypstart = hypstart + numhyp

    if hypstart + 2 * M + N * (M - 1) - 2 is not len(loghyper):
        raise RuntimeError('Incorrect number of parameters')

def omgpboundB2(loghyper, learn, covfunc, M, X, Y, Xs):
    """
    Computes the negative of the Marginalized Variational Bound (F) and its 
    derivatives wrt loghyper (dF).

    Parameters:
    loghyper: K hyperparameters, pZ, sn2 (M trajectories), logqZ
    learn: 'learnqZ', 'learnhyp', 'learnall'
    covfunc: Array of covariance functions. If it is a single one, it is shared.
    M: Number of trajectories
    X, Y, Xs: Inputs, outputs, test inputs

    (c) Miguel Lazaro-Gredilla 2010
    """

    # Initialize
    # [N, D] = X.shape
    [N, oD] = Y.shape

    logqZ = np.concatenate((np.zeros((N, 1)), loghyper[-N*(M-1):].reshape(N, M-1, order='F')))
    logqZ = logqZ - logqZ.max(1) * np.ones((1, M))
    logqZ = logqZ - np.log((np.exp(logqZ)).sum(axis=1)) * np.ones((1, M))
    qZ = np.exp(logqZ)
    sn2 = np.ones((N, 1)) * np.exp(2 * loghyper[-N*(M-1)-M:-N*(M-1)]).conj().transpose()
    logpZ = np.concatenate((np.array([0]), loghyper[-N*(M-1)-2*M+1: -N*(M-1)-M])).conj().transpose()
    logpZ = logpZ - logpZ.max(0)
    logpZ = logpZ - np.log(np.exp(logpZ).sum(axis = 0))
    logpZ = np.ones((N, 1)) * logpZ
    sqB = np.sqrt(np.divide(qZ, sn2))
    dlogqZ = np.zeros((N, M))

    # Contribution of independent (modified) GPs
    F = np.zeros((Xs.shape[0], oD, M))
    dF = np.zeros((Xs.shape[0], oD, M))

    hypstart = 1
    cm = covfunc[0]
    numhyp = # TODO: numhyp = eval(feval(cm{:})); K = feval(cm{:}, loghyper(1:numhyp), X);
    for m in M:
        if len(covfunc) is > 1:
            #TODO: cm = covfunc{m};numhyp = eval(feval(cm{:}));
            # TODO: K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
        else:
            hypstart = 1

        R = np.linalg.cholesky(np.identity(N) + K * (np.matmul(sqB[:,m], sqB[:,m].conj().transpose())).conj().transpose())
        sqBY = np.matmul(sqB[:,m], np.ones((1,oD))) * Y
        v = np.linalg.solve(R.conj().transpose(), sqBY)

        # Compute predictions
        U = np.linalg.solve(R.conj().transpose(), np.diag(sqB[:, m]))
        alpha = np.matmul(U.conj().transpose(), v)
        [Kss, Kfs] = cm(loghyper[hypstart : hypstart + numhyp - 1], X, Xs)
        F[:, :, m] = np.matmul(Kfs.conj().transpose(), alpha)

        
        dF[:, :, m] = np.matmul((sn2[0, m] + Kss - (((np.matmul(U, Kfs)) ** 2).sum(axis=0)).conj().transpose()), np.ones((1, oD)))
 
            
        hypstart = hypstart + numhyp

    if hypstart + 2 * M + N * (M - 1) - 2 is not len(loghyper):
        raise RuntimeError('Incorrect number of parameters')
