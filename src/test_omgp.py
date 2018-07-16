#!/usr/bin/python

#put omgp gen back
# fix omgp qZ random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# External modules (Python files in the same folder)
from omgp_load import *
from omgp import *
from quality import *
from parser import *

def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """
    WINDOW_MIN = 51
    WINDOW_MAX = 56
    # Number of time instants per GP, dimensions and GPs
    n = (WINDOW_MAX - WINDOW_MIN)*9
    D = 2
    M = 2

    # Tunable hyperparameters
    timescale = 20
    sigvar = 10
    noisevar = 0.0002

    # Data generation and plotting
    loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
    #[x, Y] = omgp_gen(loghyper, n, D, M)
    [x, Y, cluster_indexes, window_indexes] = parse('../inputs/log_file.txt')
    
    [x, Y] = omgp_load(x, Y, cluster_indexes, window_indexes, minSamplesxWindow=9, window_min=WINDOW_MIN, window_max=WINDOW_MAX)
    
    # x_train = x[::2]
    # Y_train = Y[::2]
    # x_test = x[1::2]
    # Y_test = Y[1::2]
    x_train = x
    Y_train = Y
    #x_test = (window_indexes[WINDOW_MAX] - window_indexes[WINDOW_MIN]) * np.random.rand(36,1) + window_indexes[WINDOW_MIN]
    x_test = np.random.randint(window_indexes[WINDOW_MIN], window_indexes[WINDOW_MAX], (36,1))
    x_test.sort(0)

    # Initialize Plot
    fig = plt.figure()
    if D == 2:
        ax = fig.add_subplot(111)
        # Changed Y_train[:, 0] to np.squeeze(np.asarray(Y_train)) when transitioning from omgp_gen to omgp_load
        ax.scatter(np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(Y_train)), c='k', marker='x')  # Add scattered points
    else: # The graph will be in 3D (dataset-> 3 dimensions or more)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train, Y_train[:, 0], Y_train[:, 1], c='k', marker='x')  # Add scattered points
        ax.set_zlabel('Z Axis')
        ax.view_init(elev=20, azim=30) #Works!
    
    ax.set_title("%d trajectories to be separated" % M)  # Add title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    plt.show()

    # OMGP tracking and plotting
    covfunc = np.matrix(['covSEiso'])     # Same type of covariance function for every GP in the model
    [F, qZ, loghyperinit, mu, C, pi0] = omgp(covfunc, M, x_train, Y_train, x_test)
    print(pi0)
    #[NMSE, NLPD] = quality(Y_test, mu, C, pi0)

    [nada, label] = qZ.max(1), qZ.argmax(1)
    fig2 = plt.figure()
    if D == 2:
        ax2 = fig2.add_subplot(111)
    else:
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_zlabel('Z Axis')
        ax2.view_init(elev=20, azim=30) #Works!

    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    plt.style.use('default')
    ax2.set_title("OMGP regression")
    
    for c in range(M):
        x_train_new = np.array([])
        Y_train0_new = np.array([])
        if D > 2:
            Y_train1_new = np.array([])
        for i in range(x_train.shape[0]):
            if (label[i] == c):
                x_train_new = np.append(x_train_new, x_train[i])
                Y_train0_new = np.append(Y_train0_new, Y_train[i, 0])
                if D > 2:
                    Y_train1_new = np.append(Y_train1_new, Y_train[i, 1])
        if D == 2:
            ax2.scatter(x_train_new, Y_train0_new, c='C%i'%c, marker='x')  # Add scattered points
            ax2.plot(np.squeeze(np.asarray(x_test)), mu[:, 0, c], c='C%i'%c)
        else:
            ax2.scatter(x_train_new, Y_train0_new, Y_train1_new, c='C%i'%c, marker='x')  # Add scattered points
            ax2.plot3D(np.squeeze(np.asarray(x_test)), mu[:, 0, c], mu[:, 1, c], c='C%i'%c)
    plt.show()
    
if __name__ == '__main__':
    test_omgp()
