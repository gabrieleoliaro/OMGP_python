#!/usr/bin/python

import numpy as np
import math

def covSEiso(loghyper, x, z):
    """
    Squared Exponential covariance function with isotropic distance measure. The 
    covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
    
    where the P matrix is ell^2 times the unit matrix and sf2 is the signal
    variance. The hyperparameters are:
    
    loghyper = [ log(ell)
                 log(sqrt(sf2)) ]
    
    For more help on design of covariance functions, try "help covFunctions".
    
    (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)
    """

    narguments = len(args)
    if narguments == 0:
        A = '2'
        return

    [n D] = np.matrix(x).shape()
    ell = math.exp(loghyper[0]);                # Characteristic length scale
    sf2 = math.exp(2 * loghyper[1])             # Signal variance

    if narguments == 2:                         
        A = sf2 * math.exp(-sq_dist(np.matrix(x).getH() / ell) / 2)
    #elif nargout == 2:                          #TODO: figure out how to convert this
        A = sf2 * np.ones(z.shape(0), 1);
        B = sf2 * math.exp(-sq_dist(np.matrix(x).getH() / ell, np.mtrix(z).getH() / ell) / 2)
    else:                                       # Compute derivative matrix
        if z == 1:                              # First parameter
            A = sf2 * math.exp(-sq_dist(np.matrix(x).getH() / ell) / 2 ). * sq_dist(np.matrix(x).getH() / ell)
        else:                                   # Second parameter
            A = 2 * sf2 * math.exp(-sq_dist(np.matrix(x).getH() / ell) / 2)


    
