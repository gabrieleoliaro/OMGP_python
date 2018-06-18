#!/usr/bin/python

import numpy as np
import math

def covSEisoCM(loghyper, x, z):
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
    [n D] = x.shape
    ell = np.exp(loghyper[0])                 # Characteristic length scale
    sf2 = np.exp(2 * loghyper[1])             # Signal variance

    A = sf2 * np.exp(-sq_dist(x.
