import numpy as np
import cv2
import scipy.io
from scipy.spatial.distance import pdist, cdist, squareform
from plotclusters3D import plotclusters3D
import matplotlib.pyplot as plt


#img = cv2.imread('m.jpg', 0)
#cv2.imshow('image',img)


mat_dict = scipy.io.loadmat('pts.mat')
data = mat_dict['data']
#data = np.array([[1, 1, 2, 2, 3, 4], [1, 1, 2, 2, 3, 4], [0, 1, 2, 0, 3, 4]])
print('data')
print(data)
print("pairwise distance:")
print(squareform(pdist(data.T)))
print("----")
print()


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
            print("Iter:", iteration)
            iteration += 1
            print("Peak:", peak)
            print()

        # Compute pairwise dist between point of interest and others.
        # Note: cdist requires its input to have the same number of dimensions.
        pair_dist = cdist(peak, data, metric).ravel()
        window = np.argwhere(pair_dist <= radius).ravel()  # Discard points outside window

        previous_peak = peak  # Store previous peak
        peak = data[window, :].mean(axis=0)  # Update peak coordinates to mean of window points
        peak = peak.reshape(1, len(peak))  # Reshape peak to (1, n)

    peak = peak.ravel()  # Reshape to (n,)

    if verbose:
        print("Converged to: ", peak, '\n')

    return peak


def mean_shift(data, r):

    data_transpose = data.T  # Note: cdist requires points as rows hence why the transpose operation
    inter_peak_threshold = r / 4  # Arbitrary threshold. See assignment description
    # Initialize matrix of peaks with peak corresponding to first index
    first_peak = find_peak(data_transpose, 0, r, verbose=False)
    peaks = first_peak[np.newaxis, :]
    labels = [0]  # Initialize first label

    for index in range(1, len(data_transpose)):  # Loop over remaining data points

        new_peak = find_peak(data_transpose, index, r, verbose=False)
        # Check if new peak should be merged with existing one
        # Note: cdist requires its input to have the same number of dimensions. Thus reshape peak from (n,) to (1, n)
        inter_peak_dist = cdist(new_peak[np.newaxis, :], peaks).ravel()
        match = np.argwhere(inter_peak_dist < inter_peak_threshold).ravel()

        if len(match) == 0:  # By construction of the matrix peaks there can only be 0 or 1 match
            peak = new_peak  # No match thus keep new peak as it
            label = labels[-1]+1  # No match thus assign new label
        else:
            new_peak = peaks[match[0], :]  # Match found thus assign matching peak to current point
            label = labels[match[0]]  # Match found thus assign matching label to current point

        peaks = np.vstack((peaks, new_peak))
        labels.append(label)

    print(peaks)
    print(peaks.shape)
    #plotclusters3D(data_transpose, labels, peaks)
    # unique = np.unique(peaks, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(data_transpose[:, 0])
    y = np.array(data_transpose[:, 1])
    z = np.array(data_transpose[:, 2])

    ax.scatter(x, y, z, marker="s", c=labels, s=40, cmap="RdBu")

    plt.show()






mean_shift(data, r=2)