#!/usr/bin/python

import numpy as np

def quality(Y, mu, C, pi0):
    [Ntst, D, M] = mu.shape

    # Make sure all input variables are numpy matrix objects
    Y = np.matrix(Y)
    mu = np.matrix(mu)
    C = np.matrix(C)
    pi0 = np.matrix(pi0)

    center = np.zeros((Ntst, D))

    for m in range(M):
        center = center + np.matmul(pi0[m], mu[:,:,m])

    NMSE = np.divide(np.mean((Y - center) ** 2), np.mean((Y - np.ones(Ntst,1) * np.mean(Y, axis=0), axis=0) ** 2, axis=0), axis=0])

    p = 0
    for m in range(M):
        p = p + pi0[m] * np.exp(np.divide(-0.5 * (Y - mu[:,:,m]) ** 2, C[:,:,m] - 0.5 * np.log(2 * math.pi * C[:,:,m])))

    NLPD = -np.mean(np.mean(np.log(p)))
