from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy


def fit_fundamental_matrix(matches):
    """
    Fit the fundamental matrix between two images given pairs of matching points.
    :param matches: (m, 4) arrays containing the coordinates of m pairwise-matching points as x, y, x', y'.
    :return: Fundamental matrix F as an (3, 3) array of rank 2.
    """

    x, y, x_prime, y_prime = matches[:, 0], matches[:, 1], matches[:, 2], matches[:, 3]  # Coordinates of matches

    A = np.array([x * x_prime,
                  x * y_prime,
                  x,
                  x_prime * y,
                  y * y_prime,
                  y,
                  x_prime,
                  y_prime,
                  np.ones_like(x)]).T

    _, _, V_transposed = np.linalg.svd(A)  # SVD decomposition of A

    F = V_transposed[-1, :].reshape(3, 3)  # Fundamental matrix corresponds to the column of V of the least sing value

    # Make the fundamental matrix rank 2
    U_f, D_f, V_f_transposed = np.linalg.svd(F)
    D_f[-1] = 0  # Set smallest s.v of F to 0
    F = U_f @ np.diag(D_f) @ V_f_transposed  # Reconstruct F as a rank 2 matrix

    return F




if __name__ == '__main__':

    # load images and match files for the first example

    I1 = Image.open('../data/library/library1.jpg')
    I2 = Image.open('../data/library/library2.jpg')
    matches = np.loadtxt('../data/library/library_matches.txt')

    N = len(matches)
    matches_to_plot = copy.deepcopy(matches)

    '''
    Display two images side-by-side with matches
    this code is to help you visualize the matches, you don't need to use it to produce the results for the assignment
    '''

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, :I1.size[0], :] = I1
    I3[:, I1.size[0]:, :] = I2
    matches_to_plot[:, 2] += I2.size[0]
    I3 = np.uint8(I3)
    I3 = Image.fromarray(I3)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I3)
    colors = iter(cm.rainbow(np.linspace(0, 1, matches_to_plot.shape[0])))
    [plt.plot([m[0], m[2]], [m[1], m[3]], color=next(colors)) for m in matches_to_plot]
    plt.show()

    # first, fit fundamental matrix to the matches
    F = fit_fundamental_matrix(matches)  # this is a function that you should write
    '''
    display second image with epipolar lines reprojected from the first image
    '''
    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I2)
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()
    fig.savefig('library2_result')

    # Plot first image and corresponding keypoints
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I1)
    ax.plot(matches[:, 0], matches[:, 1], '+r')
    plt.show()
    fig.savefig('library1_result')

    # Compute the residual (i.e MSE) of F
    p, p_prime = matches[:, 0:2], matches[:, 2:4]  # Split points
    # Append a column of 1 to the points for the evaluation of the fundamental matrix estimate
    p, p_prime = np.hstack((p, np.ones((matches.shape[0], 1)))), np.hstack((p_prime, np.ones((matches.shape[0], 1))))
    intermediate_results = p @ F
    results = np.array([intermediate_results[i, :] @ p_prime[i, :] for i in range(matches.shape[0])])
    residual = np.square(results).sum()
    print('Residual: ', residual)