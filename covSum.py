#!/usr/bin/python

import numpy as np
from covSEiso import *
from covNoise import *

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
    if covfunc[0] == covSumCM:
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype = int)
    v = np.array([], dtype = int)
    for i in range(len(covfunc)):
        if (covfunc[i] is covSEisoCM) or (covfunc[i] is covSEisoTSC) or (covfunc[i] is covSEisoDERIV):
            j = np.append(j, 2)
        elif (covfunc[i] is covNoiseCM) or (covfunc[i] is covNoiseTSC) or (covfunc[i] is covNoiseDERIV):
            j = np.append(j, 1)
            
        v = np.concatenate((v, (i * np.ones((j[i]), dtype=int))))
 
    
    
    A = np.zeros((n, n))                  # Allocate space for covariance matrix
    for i in range(len(covfunc)):
        loghyper = np.array([])
        for j in range(len(v)):
            if (v[j] == i):
                loghyper = np.append(loghyper, logtheta[j])
        A = A + covfunc[i](loghyper, x)


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
    if covfunc[0] == covSumTSC:
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype = int)
    v = np.array([], dtype = int)

    for i in range(len(covfunc)):
        if (covfunc[i] is covSEisoCM) or (covfunc[i] is covSEisoTSC) or (covfunc[i] is covSEisoDERIV):
            j = np.append(j, 2)
        elif (covfunc[i] is covNoiseCM) or (covfunc[i] is covNoiseTSC) or (covfunc[i] is covNoiseDERIV):
            j = np.append(j, 1)

        v = np.concatenate((v, (i * np.ones((j[i]), dtype=int))))
    
    alloc = np.zeros((1, 2))                          
    A = np.zeros((z.shape[0], 1))
    B = np.zeros((x.shape[0], z.shape[0]))                


    for i in range(len(covfunc)):
        loghyper = np.array([])
        for j in range(len(v)):
            if (v[j] == i):
                loghyper = np.append(loghyper, logtheta[j])
        [AA, BB] = covfunc[i](loghyper, x, z)                # Compute test covariances
        # Accumulate
        A = A + AA
        B = B + BB

def covSumDERIV(covfunc, loftheta, x, z): # TODO
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
    if covfunc[0] == covSumDERIV:
        covfunc = covfunc[1:]

    # Create and fill j array with number of parameters for each covariance function to be used in the summation
    # Create and fill v array, which indicates to which covariance parameters belong
    j = np.array([], dtype = int)
    v = np.array([], dtype = int)

    for i in range(len(covfunc)):
        if (covfunc[i] is covSEisoCM) or (covfunc[i] is covSEisoTSC) or (covfunc[i] is covSEisoDERIV):
            j = np.append(j, 2)
        elif (covfunc[i] is covNoiseCM) or (covfunc[i] is covNoiseTSC) or (covfunc[i] is covNoiseDERIV):
            j = np.append(j, 1)
        v = np.concatenate((v, (i * np.ones((j[i]), dtype=int))))


    # Compute derivative matrices
    i = v[z]                                            # Which covariance function
    j = np.sum(np.equal(v[0:z], i))                     # Which parameter in that covariance
    A = function(logtheta * np.equal(v, i), x, j)
                           
