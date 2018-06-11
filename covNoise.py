#!/usr/bin/python

import sys
import math

def covNoise(logtheta, x, z):
    """
        Independent covariance function, ie "white noise", with specified variance.
        The covariance function is specified as:

        k(x^p,x^q) = s2 * \delta(p,q)

        where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
        which is 1 iff p=q and zero otherwise. The hyperparameter is

        logtheta = [ log(sqrt(s2)) ]

        For more help on design of covariance functions, try "help covFunctions".

        (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

    """
    narguments = len(args)              # Report number of parameters
    if len(narguments == 0):              
        A = '1'
        return

    s2 = math.exp(2 * logtheta)             # Noise variance

    if (narguments == 2):                   # Compute covariance matrix
        A = s2 * np.identity(x.shape[0])
    #elif nargout == 2:                     #TODO: figure out how to convert this
        # Compute test set covariances
        A = s2
        B = 0                               # Zeros cross covariance by independence
    else                                    # Compute derivative matrix
        A = 2 * s2 * np.identity(x.shape[0])
    return [A, B]
