#!/usr/bin/python

import numpy as np
import math
from test import *
import csv

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
    ell = np.exp(loghyper[0])                 # Characteristic length scale
    sf2 = np.exp(2 * loghyper[1])             # Signal variance

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
    ell = np.exp(loghyper[0])                 # Characteristic length scale
    sf2 = np.exp(2 * loghyper[1])             # Signal variance

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
    ell = np.exp(loghyper[0])                 # Characteristic length scale

    
    sf2 = np.exp(2 * loghyper[1])             # Signal variance
    

    # Compute Derivative Matrix
    if z == 1:                              # First parameter
        A = sf2 * np.multiply(np.exp(-sq_distONE(x.conj().transpose() / ell) / 2.0), sq_distONE(x.conj().transpose() / ell))
    else:                                   # Second parameter
        A = 2.0 * sf2 * np.exp(-sq_distONE(x.conj().transpose() / ell) / 2.0)

    
    return A
        
