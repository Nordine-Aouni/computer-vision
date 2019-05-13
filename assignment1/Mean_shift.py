import numpy as np
import cv2 as cv
import scipy.io
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def find_peak(data, index, radius, metric="euclidean", threshold=0.01, search_path=True, c=4, verbose=False):
    """

    :param data: m by n array of m points in a n-dimensional space
    :param index: Row of one of the m points whose peak we'd like to find
    :param radius: Radius (in chosen metric) of the search window.
    :param metric: Distance measure (e.g 'euclidean')
    :param threshold: Distance between consecutive peaks, acts as a stopping criterion
    :param search_path: Boolean activating the search-path heuristic. Heuristic recommended for speedup.
    :param c: constant used to scale down the radius and produce distance threshold for search path heuristic.
    :param verbose: Boolean for additional printing
    :return: The peak of the corresponding point as a (n,) array
    """

    point = data[index, :]  # point of interest
    peak = point.reshape(1, len(point))  # Initialize peak and reshape to (1, n)
    previous_peak = peak + threshold+1  # Initialize dummy previous peak arbitrarily just outside threshold range

    # If search path heuristic is used, array telling which points lie in the search path. In order to speedup the search
    in_search_path = np.zeros(data.shape[0]) if search_path else None
    in_search_path_threshold = radius/c
    iteration = 1

    while np.linalg.norm(peak-previous_peak) > threshold:
        if verbose:
            print("Iter:", iteration, '\t peak: ', peak)
            iteration += 1

        # Compute pairwise dist between point of interest and others.
        # Note: cdist requires its input to have the same number of dimensions.
        pair_dist = cdist(peak, data, metric).ravel()

        if search_path:  # Heuristic returning array of points lying within search path for speedup in mean-shift.
            indices_in_search_path = np.argwhere(pair_dist <= in_search_path_threshold)  # Indices of points in path
            in_search_path[indices_in_search_path] = 1  # Store points in path

        window = np.argwhere(pair_dist <= radius).ravel()  # Create search window composed of nearby points

        previous_peak = peak  # Store previous peak
        peak = data[window, :].mean(axis=0)  # Update peak coordinates to mean of window points
        peak = peak.reshape(1, len(peak))  # Reshape peak to (1, n)

    peak = peak.ravel()  # Reshape to return a (n,) array

    if verbose:
        print("Converged to: ", peak, '\n')

    return peak, in_search_path


def mean_shift(data, radius, attraction_basin=True, search_path=True, interpeak_scaler=4):
    """
    Segment input points into group of peaks and labels.
    Find peak of every points in data according to the mean-shift algorithm.
    :param data: (n, m) array of m points in a n-dimensional space.
    :param radius: Radius of search window used to calculate peaks.
    :param attraction_basin: Boolean activating the basin-of-attraction heuristic. Heuristic recommended for speedup.
    :param search_path: Boolean activating the search-path heuristic. Heuristic recommended for speedup.
    :param interpeak_scaler: Scales down radius to define inter-peak threshold for peak merging.
    :return: (labels, peaks) Segmentation as a list of integer label and a (n, m) array of peaks corresponding to input
     points.
    """

    data_transpose = data.T  # Note: cdist requires points as rows hence why the transpose operation
    inter_peak_threshold = radius / interpeak_scaler  # Arbitrary threshold. See assignment description
    # Initialize matrix of peaks with peak corresponding to first index
    first_peak, in_search_path = find_peak(data_transpose, 0, radius, verbose=False)
    peaks = first_peak[np.newaxis, :]  # Reshape so that peak matrix is initialized shape (1, n)
    peaks_lookup = [None for _ in range(len(data_transpose))]
    labels = [0]  # Initialize first label
    label_count = 1  # Keeps track of the number of labels
    iteration = 1
    print('Mean shift running on %d pixel' % data_transpose.shape[0] )

    for index in range(1, data_transpose.shape[0]):  # Loop over remaining data points
        print('Iteration: %d/%d' % (iteration, data_transpose.shape[0]))
        iteration+=1

        if (attraction_basin or search_path) and peaks_lookup[index] is not None:  # Use look-up table if entry is available
            new_peak = peaks_lookup[index]

        else:  # No corresponding entry or no heuristics thus start search for peak

            new_peak, in_search_path = find_peak(data_transpose, index, radius, verbose=False)

            if attraction_basin:  # Basin-of-attraction heuristic: assign same peak to points lying within radius dist.
                # Note: cdist requires inputs to have same number of dimensions, and rows to stand for points.
                attraction_dist = cdist(new_peak[np.newaxis, :], data_transpose)  # dist between points and current peak
                attraction_mask = np.argwhere(attraction_dist <= radius).ravel()  # Indices of attracted points

            # Check if new peak should be merged with existing one
            # Note: cdist requires its input to have the same number of dimensions. Thus reshape peak from (n,) to (1, n)
            inter_peak_dist = cdist(new_peak[np.newaxis, :], peaks).ravel()
            match = np.argwhere(inter_peak_dist < inter_peak_threshold).ravel()

            if match.size == 0:  # By construction of the matrix peaks there can only be 0 or 1 match
                label = label_count  # No match thus assign new label and keep peak as it is.
                label_count += 1
            else:
                new_peak = peaks[match[0], :]  # Match found thus assign matching peak to current point
                label = labels[match[0]]  # Match found thus assign matching label to current point

            if attraction_basin:  # Basin-of-attraction heuristic: assign same peak to points lying within radius dist.
                for i in attraction_mask:  # For points in the basin of attraction of that peak:
                    peaks_lookup[i] = new_peak  # assign corresponding peak to that point.

        if search_path:
            indices_in_search_path = np.argwhere(in_search_path == 1).ravel()  # Indices of points in search paths
            #print("indices: ")
            #print(indices_in_search_path)
            for i in indices_in_search_path:  # For every points in search path:
                peaks_lookup[i] = new_peak  # assign corresponding peak to that point.

        peaks = np.vstack((peaks, new_peak))  # Store the peak of the point being processed
        labels.append(label)  # Store the label of the point being processed

    peaks = peaks.T  # Transpose to return a (n, m) array
    print('Num labels = ', label_count)
    return labels, peaks


def load_image_array(path, spatial_clustering=False):
    """
    Load (h, w) image from file and return it as an array.
    :param path: String path to the image.
    :param spatial_clustering: Boolean to add coordinate to color representation for more accurate segmentation.
    :return: (3, h*w) array of the image in LAB color space
    """

    img = cv.imread(path)  # Load image from file
    dimensions = (img.shape[0], img.shape[1])  # Store resolution
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Transform image to LAB color space
    dimensionality = 3  # LAB space is 3-dimensional

    if spatial_clustering:
        mask = np.ones((img.shape[0], img.shape[1]))
        coordinates = np.argwhere(mask).reshape((img.shape[0], img.shape[1], 2))
        print(coordinates)
        img = np.concatenate((img, coordinates), axis=2)
        dimensionality = 5  # LAB space + (x,y) coordinates

    # Reshape to (n, m) array of m points in n dimensions
    img = img.flatten().reshape(img.shape[0] * img.shape[1], dimensionality).T

    return img, dimensions


def segment_image(path, radius, interpeak_scaler, attraction_basin, search_path):
    """
    Segment image using the mean-shift algorithm.
    :param img: (m, n)
    :param radius: Radius of search windows used by mean-shift algorithm
    :param interpeak_scaler: Scales down radius and define inter-peak threshold for peak merging in mean-shit
    :param attraction_basin: Boolean activating the basin-of-attraction heuristic. Heuristic recommended for speedup.
    :param search_path: Boolean activating the search-path heuristic. Heuristic recommended for speedup.
    :return:
    """

    img_array, dimensions = load_image_array(path)

    t1 = time.time()
    labels, peaks = mean_shift(img_array, radius=radius, interpeak_scaler=interpeak_scaler,
                               attraction_basin=attraction_basin, search_path=search_path)
    t2 = time.time()
    print('Run time: ', t2 - t1, 'sec')
    segmentation = np.array(labels).reshape(dimensions[0], dimensions[1])

    #normalizedImg = cv.normalize(segmentation, None, 0, 255, cv.NORM_MINMAX)
    image = cv.imread(path)
    cv.imshow('Image', image)


    plt.imshow(segmentation)
    #plt.title('')
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

    #cv.imshow('dst_rt', normalizedImg)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


if __name__ == "__main__":

    file_names = ['181091.jpg', '55075.jpg', '368078.jpg']

    segment_image('181091.jpg', radius=30, interpeak_scaler=5, attraction_basin=False,
                               search_path=False)









""" 
# Runtime analysis experiments

mat_dict = scipy.io.loadmat('pts.mat')
data = mat_dict['data']

experiments = [(False, False, 'without heuristics'), (True, False, 'with 1st speed up'),
               (False, True, 'with 2nd speedup'), (True, True, 'with both speedup')]
num_iter = 100
results = []
for parameters in experiments:

    run_times = []

    for _ in range(num_iter):
        t1 = time.time()
        labels, peaks = mean_shift(data, radius=2, attraction_basin=parameters[0], search_path=parameters[1])
        t2 = time.time()
        run_times.append(t2-t1)
        #print('Run time: ', t2-t1, " seconds ",  parameters[2])

    run_times = np.array(run_times)
    results.append((run_times.mean(), run_times.std()))

fig, ax = plt.subplots()

# Example data
settings = ('No heuristics', '1st speedup', '2nd speedup', 'Both speedup')
y_pos = np.arange(len(settings))
performance = np.array([stats[0] for stats in results])
error = np.array([stats[1] for stats in results])

width = 0.35  # the width of the bars
ax.bar(y_pos, performance, width, yerr=error, align='center',
        color='blue', ecolor='red')

ax.set_xticks(y_pos)
ax.set_xticklabels(settings)
#ax.invert_yaxis()  # labels read top-to-bottom
ax.set_ylabel('Time')
ax.set_title('Runtime analysis')
plt.show()
"""
