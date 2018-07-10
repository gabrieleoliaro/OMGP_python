#!/usr/bin/python

#put omgp gen back
# fix omgp qZ random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# External modules (Python files in the same folder)
from omgp_gen import *
from omgp import *
from quality import *

def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """

    # Number of time instants per GP, dimensions and GPs

    n = 100
    D = 2
    M = 4


    # Tunable hyperparameters
    timescale = 20
    sigvar = 1
    noisevar = 0.002

    # Data generation and plotting
    loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
    [x, Y] = omgp_gen(loghyper, n, D, M)
    
    #x = read_matrix('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/x')
    #Y = read_matrix('/Users/Gabriele/Desktop/Poli/OMGP_python/inputs/Y')
    
    x_train = x[::2]
    Y_train = Y[::2]
    x_test = x[1::2]
    Y_test = Y[1::2]

    

    # Initialize Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.squeeze(np.asarray(x_train)), Y_train[:, 0], c='k', marker='x')  # Add scattered points
    ax.set_title("%d trajectories to be separated" % M)  # Add title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    plt.show()

    # OMGP tracking and plotting
    covfunc = np.matrix(['covSEiso'])     # Same type of covariance function for every GP in the model
    [F, qZ, loghyperinit, mu, C, pi0] = omgpA(covfunc, M, x_train, Y_train, x_test)
    print(pi0)
    [NMSE, NLPD] = quality(Y_test, mu, C, pi0)

    [nada, label] = qZ.max(1), qZ.argmax(1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    plt.style.use('default')
    ax2.set_title("OMGP regression")
    for c in range(M):
        x_train_new = np.array([])
        Y_train0_new = np.array([])
        for i in range(x_train.shape[0]):
            if (label[i] == c):
                x_train_new = np.append(x_train_new, x_train[i])
                Y_train0_new = np.append(Y_train0_new, Y_train[i, 0])
        ax2.scatter(x_train_new, Y_train0_new, c='C%i'%c, marker='x')  # Add scattered points
        

        ax2.plot(np.squeeze(np.asarray(x_test)), mu[:, 0, c], c='C%i'%c)
    plt.show()

    

test_omgp()
