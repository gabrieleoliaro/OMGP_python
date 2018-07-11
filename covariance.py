#!/usr/bin/python
from __future__ import division
import numpy as np
import math

from sq_dist import *


def covSEisoCM(loghyper, x):
    """
    Squared Exponential covariance function with isotropic distance measure. The
    covariance function is parameterized as:

    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)

    where the P matrix is ell^2 times the unit matrix and sf2 is the signal
    variance. The hyperparameters are:

    loghyper = [ log(ell)
                 log(sqrt(sf2)) ]

    CM stands for covariance matrix

    For more help on design of covariance functions, try "help covFunctions".

    (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)
    """
    [n, D] = np.shape(x)
    ell = np.exp(loghyper[0])  # Characteristic length scale
    sf2 = np.exp(2 * loghyper[1])  # Signal variance

    # Compute Covariance Matrix
    A = sf2 * np.exp(-sq_distONE(x.conj().transpose() / ell) / 2)

    return A


def covSEisoTSC(loghyper, x, z):
    """
    Squared Exponential covariance function with isotropic distance measure. The
    covariance function is parameterized as:

    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)

    where the P matrix is ell^2 times the unit matrix and sf2 is the signal
    variance. The hyperparameters are:

    loghyper = [ log(ell)
                 log(sqrt(sf2)) ]

    TSC stands for test set covariances

    For more help on design of covariance functions, try "help covFunctions".

    (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)
    """

    [n, D] = np.shape(x)
    ell = np.exp(loghyper[0])  # Characteristic length scale
    sf2 = np.exp(2 * loghyper[1])  # Signal variance

    # Compute Test Set Covariances
    A = sf2 * np.ones((z.shape[0], 1))
    B = sf2 * np.exp(-sq_distTWO(x.conj().transpose() / ell, z.conj().transpose() / ell) / 2)

    return [A, B]


def covSEisoDERIV(loghyper, x, z):
    """
    Squared Exponential covariance function with isotropic distance measure. The
    covariance function is parameterized as:

    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)

    where the P matrix is ell^2 times the unit matrix and sf2 is the signal
    variance. The hyperparameters are:

    loghyper = [ log(ell)
                 log(sqrt(sf2)) ]

    TSC stands for test set covariances

    For more help on design of covariance functions, try "help covFunctions".

    (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)
    """
    [n, D] = np.shape(x)
    ell = np.exp(loghyper[0])  # Characteristic length scale

    sf2 = np.exp(2 * loghyper[1])  # Signal variance

    # Compute Derivative Matrix
    if z == 1:  # First parameter
        A = sf2 * np.multiply(np.exp(-sq_distONE(x.conj().transpose() / ell) / 2.0),
                              sq_distONE(x.conj().transpose() / ell))
    else:  # Second parameter
        A = 2.0 * sf2 * np.exp(-sq_distONE(x.conj().transpose() / ell) / 2.0)

    return A


def covNoiseCM(logtheta, x):
    """
        Independent covariance function, ie "white noise", with specified variance.
        The covariance function is specified as:

        k(x^p,x^q) = s2 * \delta(p,q)

        where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
        which is 1 iff p=q and zero otherwise. The hyperparameter is

        logtheta = [ log(sqrt(s2)) ]

        CM stands for Covariance matrix

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

    """

    s2 = np.exp(2 * logtheta)             # Noise variance
    A = s2 * np.identity(x.shape[0])
    return A

def covNoiseTSC(logtheta):
    """
        Independent covariance function, ie "white noise", with specified variance.
        The covariance function is specified as:

        k(x^p,x^q) = s2 * \delta(p,q)

        where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
        which is 1 iff p=q and zero otherwise. The hyperparameter is

        logtheta = [ log(sqrt(s2)) ]

        TSC stands for test set covariances

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

    """

    s2 = np.exp(2 * logtheta)             # Noise variance

    # Compute test set covariances
    A = s2
    B = 0                                   # Zero cross covariances by independence

    return [A, B]

def covNoiseDERIV(logtheta, x):
    """
        Independent covariance function, ie "white noise", with specified variance.
        The covariance function is specified as:

        k(x^p,x^q) = s2 * \delta(p,q)

        where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
        which is 1 iff p=q and zero otherwise. The hyperparameter is

        logtheta = [ log(sqrt(s2)) ]

        DERIV stands for derivative matrix

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

    """

    s2 = np.exp(2 * logtheta)             # Noise variance
    A = 2 * s2 * np.identity(x.shape[1])


def covSumCM(covfunc, logtheta, x):
    """
        covSum - compose a covariance function as the sum of other covariance
        functions. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work.

        CM stands for Covariance Matrix

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.
    """

    [n, D] = np.shape(x)

    # Pop the 'covSumCM' function out of the covfunc list, so that we use the array to compute the subcovariances
    if covfunc[0] == 'covSum':
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype=int)
    v = np.array([], dtype=int)
    for i in range(len(covfunc)):
        if covfunc[i] == 'covSEiso':
            j = np.append(j, 2)
        elif covfunc[i] == 'covNoise':
            j = np.append(j, 1)
        v = np.concatenate((v, ((i + 1) * np.ones((j[i]), dtype=int))))

    A = np.zeros((n, n))  # Allocate space for covariance matrix
    for i in range(len(covfunc)):
        loghyper = np.array([])
        for j in range(len(v)):
            if (v[j] == i + 1):
                loghyper = np.append(loghyper, logtheta[j])
        if covfunc[i] == 'covSEiso':
            A = A + covSEisoCM(loghyper, x)
        elif covfunc[i] == 'covNoise':
            A = A + covNoiseCM(loghyper, x)

    return A


def covSumTSC(covfunc, loftheta, x, z):
    """
        covSum - compose a covariance function as the sum of other covariance
        functions. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work.

        TSC stands for Test Set Covariances

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.
    """

    [n, D] = np.shape(x)

    # Pop the 'covSumCM' function out of the covfunc list, so that we use the array to compute the subcovariances
    if covfunc[0] == 'covSum':
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype=int)
    v = np.array([], dtype=int)

    for i in range(len(covfunc)):
        if covfunc[i] == 'covSEiso':
            j = np.append(j, 2)
        elif covfunc[i] == 'covNoise':
            j = np.append(j, 1)

        v = np.concatenate((v, ((i + 1) * np.ones((j[i]), dtype=int))))

    alloc = np.zeros((1, 2))
    A = np.zeros((z.shape[0], 1))
    B = np.zeros((x.shape[0], z.shape[0]))

    for i in range(len(covfunc)):
        loghyper = np.array([])
        for j in range(len(v)):
            if (v[j] == i):
                loghyper = np.append(loghyper, logtheta[j])
        if covfunc[i] == 'covSEiso':
            [AA, BB] = covSEisoTSC[i](loghyper, x, z)  # Compute test covariances
        elif covfunc[i] == 'covNoise':
            [AA, BB] = covNoiseTSC[i](loghyper, x, z)  # Compute test covariances

        # Accumulate
        A = A + AA
        B = B + BB


def covSumDERIV(covfunc, loftheta, x, z):
    """
        covSum - compose a covariance function as the sum of other covariance
        functions. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work.

        DERIV stands for derivative matrix

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.
    """

    [n, D] = np.shape(x)

    # Pop the 'covSumCM' function out of the covfunc list, so that we use the array to compute the subcovariances
    if covfunc[0] == 'covSum':
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype=int)
    v = np.array([], dtype=int)

    for i in range(len(covfunc)):
        if covfunc[i] == 'covSEiso':
            j = np.append(j, 2)
        elif covfunc[i] == 'covNoise':
            j = np.append(j, 1)
        v = np.concatenate((v, ((i + 1) * np.ones((j[i]), dtype=int))))

    # Compute derivative matrices
    i = v[z]  # Which covariance function
    j = np.sum(np.equal(v[0:z], i))  # Which parameter in that covariance
    A = function(logtheta * np.equal(v, i), x, j)
