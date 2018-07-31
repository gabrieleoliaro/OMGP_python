#!/usr/bin/python

from omgp import *
# External modules (Python files in the same folder)
from omgp_load import *

OMGP_GENERATE = 2
OMGP_LOAD = 1



def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """

    # Number of time instants per GP, dimensions and GPs
    n = 100         # number of points x GP
    D = 2           # dimensions
    M = 3           # number of GPs/trajectories

    # Parameter telling the program whether to load data from a log file or generate random values
    omgp_mode = OMGP_LOAD
    if omgp_mode == OMGP_LOAD:
        assert(D == 2)

    # Tunable hyperparameters
    timescale = 20
    sigvar = 1
    noisevar = 0.0002

    if omgp_mode == OMGP_GENERATE:
        # Data generation and plotting
        loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
        [x_train, Y_train, x_test, Y_test] = omgp_gen(loghyper, n, D, M)
    elif omgp_mode == OMGP_LOAD:
        # load windows
        [x_train, Y_train, x_test] = omgp_load('../inputs/log_file.txt')

 

    ###### show data as scatter plot #####
    fig = plt.figure()
    if D == 2:
        ax = fig.add_subplot(111)
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
    ##### --- #####

    ###### OMGP tracking and plotting #####
    covfunc = np.matrix(['covSEiso'])     # Same type of covariance function for every GP in the model
    [F, qZ, loghyperinit, mu, C, pi0] = omgp(covfunc, M, x_train, Y_train, x_test)
    print(pi0)
    if omgp_mode == OMGP_LOAD:
        [NMSE, NLPD] = quality(Y_test, mu, C, pi0)

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
