#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import omgp_gen

def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """

    # Number of time instants per GP, dimensions and GPs

    n = 100
    D = 2
    M = 3

    # Tunable hyperparameters
    timescale = 20
    sigvar = 1
    noisevar = 0.002

    # Data generation and plotting
    loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
    [x, Y] = omgp_gen(loghyper, n, D, M)

    x_train = x[::2]
    Y_train = Y[::2,:]
    x_test = x[1::2]
    Y_test = Y[1::2,:]

    #Initialize Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train, Y_train[:,0], Y_train[:,1], 'k', 'x')   # Add scattered points
    ax.set_title("%d trajectories to be separated (drag to see)" % M) # Add title

    plt.show()
