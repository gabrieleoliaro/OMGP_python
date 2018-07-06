#!/usr/bin/python

import numpy as np
from test import *
import csv

def quality(Y, mu, C, pi0):
    [Ntst, D, M] = mu.shape
    
 
    center = np.zeros((Ntst, D))

    for m in range(M):
        center = center + np.multiply(pi0[m], mu[:,:,m])

    NMSE = np.mean(np.divide(np.mean(np.power((Y-center) , 2)), np.mean(np.power((Y - np.ones((Ntst, 1)) * np.mean(Y) ), 2))))
    

    p = 0
    for m in range(M):
        p = p + pi0[m] * np.exp(np.divide(-0.5 * np.power((Y - mu[:, :, m]), 2), C[:, :, m]) - 0.5 * np.log(2 * np.pi * C[:, :, m]))
    
                
    NLPD = -np.mean(np.mean(np.log(p)))
    return [NMSE, NLPD]
