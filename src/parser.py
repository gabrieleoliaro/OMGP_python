#!/usr/bin/python

import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# This code assumes that the clusters are written down in a sorted fashion
# np.set_printoptions(threshold=np.nan)


def parse(filename):
    # Read all data (in CSV format) from log file

    x = np.array([], dtype=int)
    Y = np.array([])
    cluster_indexes = np.array([], dtype=int)
    window_indexes = np.array([],
                              dtype=int)  # Array representing at which sample number (from the beginning) the ith window begins

    f_in = open(filename, 'r')
    reader = csv.reader(f_in, delimiter=",")

    additional_offset = 0
    offset = 0
    for i, line in enumerate(reader):
        if line == ['--']:
            window_indexes = np.append(window_indexes, offset)
            offset += additional_offset + 1
            additional_offset = 0
        elif line != []:
            additional_offset = max(additional_offset, int(line[1]))
            if len(cluster_indexes) > int(line[
                                              0]) + 1:  # There's already an element of number line[0]+1 in x, so you can insert line[1] right before it
                x = np.insert(x, cluster_indexes[int(line[0]) + 1], int(line[1]) + offset)
                Y = np.insert(Y, cluster_indexes[int(line[0]) + 1], float(line[2]))
                for index, element in enumerate(cluster_indexes):
                    if index >= int(line[0]) + 1:
                        cluster_indexes[index] = element + 1
            else:  # You can't insert line[1] right before the first element of number line[0]+1, because there's no such element
                if int(line[0]) == len(
                        cluster_indexes):  # If this is the first element characterized by the current number, you have to save its address into cluster_indexes
                    cluster_indexes = np.append(cluster_indexes, len(x))
                x = np.append(x, int(line[1]) + offset)
                Y = np.append(Y, float(line[2]))
    f_in.close()

    return [x, Y, cluster_indexes, window_indexes]



def plot(x, Y, cluster_indexes, window_indexes, clust_min=0, clust_max=np.inf, window_min=0, window_max=np.inf, minSamplesxWindow=0, maxSamplesxWindow=np.inf):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Angle')

    if window_min < 0 or window_min >= len(window_indexes):
        window_min = 0
    if window_max < 0 or window_max >= len(window_indexes):
        window_max = len(window_indexes) - 1
    if clust_min < 0 or clust_min >= len(cluster_indexes):
        clust_min = 0
    if clust_max < 0 or clust_max >= len(cluster_indexes):
        clust_max = len(cluster_indexes) - 1
    if minSamplesxWindow < 0 or minSamplesxWindow > 9:
        minSamplesxWindow = 0
    if maxSamplesxWindow < 0 or maxSamplesxWindow > 9:
        maxSamplesxWindow = 9

    for i in range(len(cluster_indexes)):
        x_input = np.array([], dtype=int)
        y_input = np.array([])
        if i < clust_min or i > clust_max:
            continue
        for w in range(window_min, window_max):
            sample_min = window_indexes[w]
            sample_max = window_indexes[window_max] if w == window_max-1 else window_indexes[w+1]
            if i == len(cluster_indexes) - 1:
                # Go from here to end of x/Y array
                x_plot = np.array([], dtype=int)
                Y_plot = np.array([])
                for j in range(cluster_indexes[i], len(x)):
                    if x[j] >= sample_min and x[j] < sample_max:
                        x_plot = np.append(x_plot, x[j])
                        Y_plot = np.append(Y_plot, Y[j])
                if len(x_plot) >= minSamplesxWindow and len(x_plot) <= maxSamplesxWindow:
                    ax.plot(x_plot, Y_plot, c='C%i' % i)
                    ax.scatter(x_plot, Y_plot, c='C%i' % i, marker='x')
                    x_input = np.append(x_input, x_plot)
                    y_input = np.append(y_input, Y_plot)
            else:
                # Go from here to cluster_indexes[i+1]
                x_plot = np.array([], dtype=int)
                Y_plot = np.array([])
                for j in range(cluster_indexes[i], cluster_indexes[i+1]):
                    if x[j] >= sample_min and x[j] < sample_max:
                        x_plot = np.append(x_plot, x[j])
                        Y_plot = np.append(Y_plot, Y[j])
                if len(x_plot) >= minSamplesxWindow and len(x_plot) <= maxSamplesxWindow:
                    ax.plot(x_plot, Y_plot, c='C%i' % i)
                    ax.scatter(x_plot, Y_plot, c='C%i' % i, marker='x')
                    x_input = np.append(x_input, x_plot)
                    y_input = np.append(y_input, Y_plot)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    [x, Y, cluster_indexes, window_indexes] = parse('../inputs/log_file.txt')
    window_counter = len(window_indexes)
    print('window counter: ', window_counter)
    M = len(cluster_indexes)
    plot(x, Y, cluster_indexes, window_indexes, minSamplesxWindow=9, window_min=51, window_max=56)
    print(cluster_indexes)
    print('M:', M)
