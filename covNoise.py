#!/usr/bin/python

import math

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

    s2 = math.exp(2 * logtheta)             # Noise variance
    A = s2 * np.identity(x.shape[1])

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

    s2 = math.exp(2 * logtheta)             # Noise variance

    # Compute test set covariances
    A = s2
    B = 0                                   # Zero cross covariances by independence

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

    s2 = math.exp(2 * logtheta)             # Noise variance
    A = 2 * s2 * np.identity(x.shape[1])
