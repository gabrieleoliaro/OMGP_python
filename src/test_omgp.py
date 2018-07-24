#!/usr/bin/python

#put omgp gen back
# fix omgp qZ random

from omgp import *
# External modules (Python files in the same folder)
from omgp_load import *
from parser import *


def print_windows(windows, start_from=0, n_win=16, min_pts=9):
    """Plot a sequence of windows
    Parameters:
        windows: the list of windows
        start_from: index of first window to be plotted.
        n_win: how many windows to plot
        min_pts: the minimum number of points to be plotted.
    """
    fig = plt.figure()
    for i, w in enumerate(range(start_from, start_from + n_win)):
        l = int(np.sqrt(n_win))
        ax = fig.add_subplot(l, l, i + 1)
        for c in windows[w].values():
            if len(c) >= min_pts:
                ax.plot(c)
        ax.set_ylim((89, 181))
        ax.set_xlabel('sample')
        ax.set_ylabel('angle')
        ax.set_yticks(range(90, 180, 20))
        ax.set_xticks(range(0, 10, 3))
        ax.grid()

    plt.tight_layout()
    plt.show()


def test_omgp():
    """
    Tests functions omgp_gen and omgp
    """
    WINDOW_MIN = 1
    WINDOW_MAX = 6

    # Number of time instants per GP, dimensions and GPs
    #n = (WINDOW_MAX - WINDOW_MIN)*9
    n = 100
    D = 2
    M = 2

    # Tunable hyperparameters
    timescale = 20
    sigvar = 1
    noisevar = 0.0002

    # Data generation and plotting
    # loghyper = np.array([np.log(timescale), 0.5 * np.log(sigvar), 0.5 * np.log(noisevar)])
    # [x, Y] = omgp_gen(loghyper, n, D, M)

    # load windows
    windows = new_parse('../inputs/log_file.txt')

    # plotting some windows for inspection
    print_windows(windows, start_from=0)

    # get windows with at least 9 points
    Y = []
    for w in windows:
        for c in w.values():
            if len(c) >= 9:
                Y.append(c)
    # ---

    # plotting lines with at least 9 points
    fig = plt.figure()
    for y in Y:
        plt.plot(y)
    plt.title("all lines with at least 9 samples")  # Add title
    plt.xlabel('sample')
    plt.ylabel('angle')
    plt.show()
    # ---

    # setting the data set properly
    x_train = np.array([], dtype=int)
    for y in Y:
        x_train = np.append(x_train, range(len(y)))
    x_train = np.matrix(x_train).conj().transpose()
    Y_train = np.array([item for sublist in Y for item in sublist])
    Y_train = np.matrix(Y_train).conj().transpose()
    x_test = np.random.randint(0,9, (9,1))
    x_test.sort(0)

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
