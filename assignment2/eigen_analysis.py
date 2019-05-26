import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_faces(path, key='fea'):
    mat_data = loadmat(path)
    faces = mat_data[key]

    return faces


path = 'faces/ORL_32x32.mat'
X = load_faces(path)  # Instance are on rows, features in columns

X = X.T  # We want features on rows and observations as columns
N = X.shape[1]
X = X / 255  # Normalize pixel intensities to [0, 1]
original_images = np.copy(X)
X_mean = X.mean(axis=1)[:, np.newaxis]  # Average of each feature
X = X - X_mean  # Center features around their mean
X_copy = np.copy(X)

T = (1/N) * (X.T @ X)  # Artificial matrix to facilitate eigenvectors computations
eigenvalues, e = np.linalg.eig(T)  # Find eigenvalues of the covariance matrix
eigenvectors = X @ e  # Finds the eigenvectors of the covariance matrix

index_array = eigenvalues.argsort()[::-1]  # Indices of eigenvalues sorted in decreasing order
k = 30  # Number of principal components to keep
princ_comps = eigenvectors[:, index_array[:k]].T  # Principal components as rows

norms = np.linalg.norm(princ_comps, axis=1)  # Normalize PC's
princ_comps = princ_comps / norms[:, np.newaxis]


"""
# plot some eigenfaces
eigenface = princ_comps[0, :].reshape(32, 32).T  # Plot one eigen face

fig, axarr = plt.subplots(3, 3)
fig_count = 0

for i in range(axarr.shape[0]):
    for j in range(axarr.shape[0]):
        axarr[i][j].imshow(princ_comps[fig_count, :].reshape(32, 32).T, cmap='gray')
        fig_count += 1
plt.show()
"""

weights = princ_comps @ X  # column i contains all the weights of image i
print(weights[:, 0].min(), weights[:, 0].max(), weights[:, 0].mean() )

""" 
# plot an image and corresponding main and laest eigenface
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

print(princ_comps.shape)
reconstruction = X_mean + (princ_comps.T @ weights)
# Normalize reconstructed values back to [0, 1]
#reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max()-reconstruction.min())

# Plot an original image and its reconstruction side by side
img_id = 50
original_img = original_images[:, img_id].reshape(32, 32).T
reconstructed_img = reconstruction[:, img_id].reshape(32, 32).T
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(original_img, cmap='gray')
axarr[0].title.set_text('Original image')
axarr[1].imshow(reconstructed_img, cmap='gray')
axarr[1].title.set_text("Reconstructed image")
plt.savefig('eigenface')
plt.show()


print('----')
print("Original image:")
print(original_img)
print('\nReconstructed image:')
print(reconstructed_img)
