import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.spatial import distance


class eigenface:
    def __init__(self, path, split, k):

        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split_data(path, split)  # Observations in columns, features on columns

        N = self.X_train.shape[1]  # Number of observations
        self.X_train = self.X_train / 255  # Normalize pixel intensities to [0, 1]
        original_images = np.copy(self.X_train)
        self.X_mean = self.X_train.mean(axis=1)[:, np.newaxis]  # Average of each feature
        self.X_train = self.X_train - self.X_mean  # Center features around their mean

        plt.imshow(self.X_mean.reshape(32, 32).T, cmap='gray')
        plt.axis('off')
        plt.savefig('mean_face', bbox_inches='tight')

        T = (1/N) * (self.X_train.T @ self.X_train)  # Artificial matrix to facilitate eigenvectors computations
        eigenvalues, e = np.linalg.eig(T)  # Find eigenvalues of the covariance matrix
        eigenvectors = self.X_train @ e  # Finds the eigenvectors of the covariance matrix

        index_array = eigenvalues.argsort()[::-1]  # Indices of eigenvalues sorted in decreasing order
        self.princ_comps = eigenvectors[:, index_array[:k]].T  # Principal components as rows

        norms = np.linalg.norm(self.princ_comps, axis=1)  # Normalize PC's
        self.princ_comps = self.princ_comps / norms[:, np.newaxis]

        #self.plot_eigenfaces(self.princ_comps, split, k, 8, 5)

        self.weights = self.princ_comps @ self.X_train  # column i contains all the weights of image i
        reconstruction = self.X_mean + (self.princ_comps.T @ self.weights)  # Reconstruct a training image
        #self.plot_reconstruction(original_images, reconstruction, split, k, img_id=0)


    def load_and_split_data(self, path, split, key='fea'):
        matlab_dict = loadmat(path)

        X = matlab_dict[key]
        X = X.T  # We want observation in columns and features as rows

        y = matlab_dict['gnd'].flatten()

        if split == 3:  # Set right file
            split_path = 'faces/3Train/3.mat'
        elif split == 5:
            split_path = 'faces/5Train/5.mat'
        else:
            split_path = 'faces/7Train/7.mat'


        matlab_dict = loadmat(split_path)  # Load file containing train/test split

        train_indices, test_indices = matlab_dict['trainIdx'], matlab_dict['testIdx']
        train_indices = train_indices.flatten()
        test_indices = test_indices.flatten()

        # Substract 1 as MATLAB starts indexing from 1 while python starts from 0
        train_indices -= 1
        test_indices -= 1

        X_train, X_test = X[:, train_indices], X[:, test_indices]  # Split data into train and test set
        y_train, y_test = y[train_indices], y[test_indices]


        return X_train, X_test, y_train, y_test

    def plot_eigenfaces(self, princ_comps, split, k, num_row, num_col):
        """Plot eigenfaces"""

        fig, axarr = plt.subplots(num_row, num_col, figsize=(20, 20))
        fig_count = 0
        for i in range(axarr.shape[0]):
            for j in range(axarr.shape[1]):
                axarr[i][j].imshow(princ_comps[fig_count, :].reshape(32, 32).T, cmap='gray', aspect="auto")
                axarr[i][j].axis('off')
                fig_count += 1
        plt.subplots_adjust(wspace=.05, hspace=.05)
        title = 'top_%d_faces_split_%d' % (k, split)
        plt.savefig(title, bbox_inches='tight')
        plt.show()

    def plot_reconstruction(self, original_images, reconstruction, split, k, img_id):
        # Plot an original image and its reconstruction side by side
        original_img = original_images[:, img_id].reshape(32, 32).T
        reconstructed_img = reconstruction[:, img_id].reshape(32, 32).T
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(original_img, cmap='gray')
        axarr[0].title.set_text('Original image')
        axarr[1].imshow(reconstructed_img, cmap='gray')
        axarr[1].title.set_text("Reconstructed image")
        title = 'reconstruction_split_%d_k_%d_id_%d' % (split, k, img_id)
        plt.savefig(title, bbox_inches='tight')
        plt.show()


    def predict(self):
        X_test_mean = self.X_test.mean(axis=1)[:, np.newaxis]  # Average of each feature
        self.X_test = self.X_test - X_test_mean
        test_weights = self.princ_comps @ self.X_test

        pairwise_dist = distance.cdist(test_weights.T, self.weights.T)
        nearest_neighbors = pairwise_dist.argmin(axis=1)

        score = (self.y_test == self.y_train[nearest_neighbors]).astype(int)
        accuracy = score.sum()/score.size
        print('accuracy = ', accuracy)





path = 'faces/ORL_32x32.mat'
split = 7
k = 100
eigenface = eigenface(path, split, k)
eigenface.predict()

""" 
# plot an image and corresponding main and least eigenface
img_id = 0
original_img = original_images[:, img_id].reshape(32, 32).T
f, axarr = plt.subplots(1,3)
axarr[0].title.set_text('Original image')
axarr[0].imshow(original_img, cmap='gray')
axarr[1].title.set_text("Main eigenface")
axarr[1].imshow(princ_comps[weights[:, img_id].argmax(), :].reshape(32, 32).T, cmap='gray')
axarr[2].title.set_text("Least eigenface")
axarr[2].imshow(princ_comps[weights[:, img_id].argmin(), :].reshape(32, 32).T, cmap='gray')
plt.show()
"""


""" 
reconstruction = X_mean + (princ_comps.T @ weights)
# Normalize reconstructed values back to [0, 1]
#reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max()-reconstruction.min())
"""


