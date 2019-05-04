import numpy as np
import cv2
import scipy.io
from scipy.spatial.distance import pdist, cdist, squareform
from plotclusters3D import plotclusters3D
import matplotlib.pyplot as plt
import time


def find_peak(data, index, radius, metric="euclidean", threshold=0.01, verbose=False):
    """

    :param data: m by n array of m points in a n-dimensional space
    :param index: Row of one of the m points whose peak we'd like to find
    :param radius: Radius (in chosen metric) of the search window.
    :param metric: Distance measure (e.g 'euclidean')
    :param threshold: Distance between consecutive peaks, acts as a stopping criterion
    :param verbose: Boolean for additional printing
    :return: The peak of the corresponding point as a (n,) array
    """

    point = data[index, :]  # point of interest
    peak = point.reshape(1, len(point))  # Initialize peak and reshape to (1, n)
    previous_peak = peak + threshold+1  # Initialize dummy previous peak now to begin the search
    iteration = 1

    while np.linalg.norm(peak-previous_peak) > threshold:
        if verbose:
            print("Iter:", iteration, '\t peak: ', peak)
            iteration += 1

        # Compute pairwise dist between point of interest and others.
        # Note: cdist requires its input to have the same number of dimensions.
        pair_dist = cdist(peak, data, metric).ravel()
        window = np.argwhere(pair_dist <= radius).ravel()  # Discard points outside window

        previous_peak = peak  # Store previous peak
        peak = data[window, :].mean(axis=0)  # Update peak coordinates to mean of window points
        peak = peak.reshape(1, len(peak))  # Reshape peak to (1, n)

    peak = peak.ravel()  # Reshape to return a (n,) array

    if verbose:
        print("Converged to: ", peak, '\n')

    return peak


def mean_shift(data, radius, attraction_basin=True):
    """
    Find peak of every points in data according to the mean-shift algorithm.
    :param data: (m, n) array of m points in a n-dimensional space
    :param radius: Radius of search window used to calculate peaks.
    :param attraction_basin: Boolean activating the basin-of-attraction heuristic. Heuristic recommended for speedup.
    :return: (labels, peaks) Segmentation as a list of integers and a (m, n) array of peaks corresponding to input points
    """

    data_transpose = data.T  # Note: cdist requires points as rows hence why the transpose operation
    inter_peak_threshold = radius / 4  # Arbitrary threshold. See assignment description
    # Initialize matrix of peaks with peak corresponding to first index
    first_peak = find_peak(data_transpose, 0, radius, verbose=False)
    peaks = first_peak[np.newaxis, :]  # Reshape so that peak matrix is initialized shape (1, n)
    peaks_lookup = [None for _ in range(len(data_transpose))]
    labels = [0]  # Initialize first label

    for index in range(1, len(data_transpose)):  # Loop over remaining data points

        if attraction_basin and peaks_lookup[index] is not None:  # Use look-up table if entry is available
            new_peak = peaks_lookup[index]

        else:  # No corresponding entry or no heuristics thus start search for peak

            new_peak = find_peak(data_transpose, index, radius, verbose=False)
            # Check if new peak should be merged with existing one
            # Note: cdist requires its input to have the same number of dimensions. Thus reshape peak from (n,) to (1, n)
            inter_peak_dist = cdist(new_peak[np.newaxis, :], peaks).ravel()
            match = np.argwhere(inter_peak_dist < inter_peak_threshold).ravel()

            if len(match) == 0:  # By construction of the matrix peaks there can only be 0 or 1 match
                label = labels[-1]+1  # No match thus assign new label and peak as it is.
            else:
                new_peak = peaks[match[0], :]  # Match found thus assign matching peak to current point
                label = labels[match[0]]  # Match found thus assign matching label to current point

        if attraction_basin:  # Basin-of-attraction heuristic: assign same peak to points lying within radius dist.
            # Note: cdist requires inputs to have same number of dimensions and rows to be points.
            attraction_dist = cdist(new_peak[np.newaxis, :], data_transpose)  # dist between points and current peak
            attraction_mask = np.argwhere(attraction_dist <= radius).ravel()  # Indices of attracted points
            if len(attraction_mask) > 0:
                for i in attraction_mask:
                    peaks_lookup[i] = new_peak

        peaks = np.vstack((peaks, new_peak))
        labels.append(label)

    return labels, peaks


#img = cv2.imread('m.jpg', 0)
#cv2.imshow('image',img)

mat_dict = scipy.io.loadmat('pts.mat')
data = mat_dict['data']

t1 = time.time()
labels, peaks = mean_shift(data, radius=2)
t2 = time.time()
print('Run time: ', t2-t1, " seconds with heuristic")

t1 = time.time()
labels, peaks = mean_shift(data, radius=2, attraction_basin=False)
t2 = time.time()
print('Run time: ', t2-t1, " seconds without heuristic")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data[0, :])
y = np.array(data[1, :])
z = np.array(data[2, :])
ax.scatter(x, y, z, marker="s", c=labels, s=40, cmap="RdBu")
plt.show()