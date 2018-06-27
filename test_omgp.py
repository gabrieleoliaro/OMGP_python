#!/usr/bin/python

#put omgp gen back
# fix omgp qZ random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test import *

# External modules (Python files in the same folder)
from omgp_gen import *
#from omgp import *
from covSEiso import *

def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """

    # Number of time instants per GP, dimensions and GPs

    n = 20
    D = 2
    M = 3


    # Tunable hyperparameters
    timescale = 20
    sigvar = 1
    noisevar = 0.002

    # Data generation and plotting
    loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
 #   [x, Y] = omgp_gen(loghyper, n, D, M)
    x = read_matrix('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/x')
    Y = read_matrix('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/Y')
    
    x_train = x[::2]
    Y_train = Y[::2]
    x_test = x[1::2]
    Y_test = Y[1::2]


    # Initialize Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train, Y_train[:, 0], Y_train[:, 1], c='k', marker='x')  # Add scattered points
    ax.set_title("%d trajectories to be separated (drag to see)" % M)  # Add title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

    # OMGP tracking and plotting
    covfunc = np.matrix(('covSEiso'))     # Same type of covariance function for every GP in the model
 #   [F, qZ, loghyperinit, mu, C, pi0] = omgpA(covfunc, M, x_train, Y_train, x_test)
    #print(pi0)
    #[NMSE, NLPD] = quality(Y_test, mu, C, pi0)
    

test_omgp()
