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
    window_indexes = np.array([], dtype=int) # Array representing at which sample number (from the beginning) the ith window begins

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


# def filter_angles(x, Y, min_angle=0., max_angle=180.):
#     x2 = np.array([], dtype=int)
#     Y2 = np.array([])
#     for i in range(len(Y)):
#         if Y[i] >= min_angle and Y[i] <= max_angle:
#             x2 = np.append(x2, x[i])
#             Y2 = np.append(Y2, Y[i])
#     x = x2
#     Y = Y2
#
#     return [x, Y]


def plot(x, Y, cluster_indexes, window_indexes, clust_min=0, clust_max=np.inf, window_min=0, window_max=np.inf):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Angle')

    # Make sure clust_min, clust_max, window_min and window_max are in the proper range
    if clust_min < 0 or clust_min > len(cluster_indexes):
        clust_min = 0
    if clust_max < 0 or clust_max > len(cluster_indexes):
        clust_max = len(cluster_indexes)
    if window_min < 0 or window_min > len(window_indexes):
        window_min = 0
    if window_max < 0 or window_max > len(window_indexes):
        window_max = len(window_indexes) -1

    for i in range(len(cluster_indexes)):
        if i < clust_min or i > clust_max:
            continue
        if i == len(cluster_indexes) - 1:
            # Go from here to end of x/Y array
            if len(x[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])]) == 9: #Only print clusters that appear in 9 samples in the current window
                ax.plot(x[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])], Y[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])], c='C%i' % i)
                ax.scatter(x[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])], Y[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])], c='C%i' % i, marker='x')
            print('samples for cluster %i: ' %i, len(x[cluster_indexes[i] + window_indexes[window_min] : min(len(x), cluster_indexes[i] + window_indexes[window_max] - window_indexes[window_max-1])]))
        else:
            # Go from here to cluster_indexes[i+1]
            #TODO: add the - window_indexes[window_max-1] but only if window_max-1 >= 0 part
            if len(x[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])]) == 9:  # Only print clusters that appear in 9 samples in the current window
                ax.plot(x[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])], Y[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])], c='C%i' % i)
                ax.scatter(x[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])], Y[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])], c='C%i' % i, marker='x')
            print('samples for cluster %i: ' %i, len(x[cluster_indexes[i] + window_indexes[window_min] : min(cluster_indexes[i + 1], cluster_indexes[i] + window_indexes[window_max])]))
    ax.grid(True)
    plt.show()


# # Print all local variabels to file
# f = open("variables.csv", "w")
# w = csv.writer(f)
# local_variables = locals().copy()
# for key, value in local_variables.items():
#     w.writerow([key, value])
# f.close()

if __name__ == "__main__":

    [x, Y, cluster_indexes, window_indexes] = parse('../inputs/log_file.txt')
    window_counter = len(window_indexes)
    M = len(cluster_indexes)
    #[x, Y] = filter_angles(x, Y)
    plot(x, Y, cluster_indexes, window_indexes, window_min=0, window_max=1)
    print(cluster_indexes)
    print('M:', M)

