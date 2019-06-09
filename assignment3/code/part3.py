import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def camera_centre(camera):
    U, S, V = np.linalg.svd(camera)
    return V[-1,:] / V[-1,-1]

option = "library"
option = "house"

with open(option+"_matches.txt", "r") as f:
    matches = np.array([[float(x) for x in line.split()] for line in f])

with open(option+"1_camera.txt", "r") as f:
    P1 = np.array([[float(x) for x in line.split()] for line in f])

with open(option+"2_camera.txt", "r") as f:
    P2 = np.array([[float(x) for x in line.split()] for line in f])

p31 = P1[2,:]
p21 = P1[1,:]
p11 = P1[0,:]
c1 = camera_centre(P1)

p32 = P2[2,:]
p22 = P2[1,:]
p12 = P2[0,:]
c2 = camera_centre(P2)

x1 = matches[:,0]
y1 = matches[:,1]
x2 = matches[:,2]
y2 = matches[:,3]

# PLOT
fig = plt.figure()
fig.set_size_inches(12.5, 8.5)

ax = fig.add_subplot(221)
ax.imshow(plt.imread(option+"1.jpg"))
ax = fig.add_subplot(222)
ax.imshow(plt.imread(option+"1.jpg"))
ax.scatter(x1, y1)

ax = fig.add_subplot(223)
ax.imshow(plt.imread(option+"2.jpg"))
ax = fig.add_subplot(224)
ax.imshow(plt.imread(option+"2.jpg"))
ax.scatter(x2, y2)


points = []
for i in range(matches.shape[0]):
    A = np.array([x1[i] * p31 - p11,
                  y1[i] * p31 - p21,
                  x2[i] * p32 - p12,
                  y2[i] * p32 - p22])
    U, S, V = np.linalg.svd(A)
    k = np.argmin(S)
    X = V[k,:] # Row wise since we technically have V transpose
    points.append(X)

points = np.array(points)

for col in range(points.shape[1]):
    points[:,col] = points[:,col] /points[:,-1]

if option is "house":
    x = -points[:, 2]
    y = points[:, 0]
    z = points[:, 1]
    # Camera centers
    cx1 = -c1[2]
    cy1 = c1[0]
    cz1 = c1[1]
    cx2 = -c2[2]
    cy2 = c2[0]
    cz2 = c2[1]
else:
    x = points[:, 2]
    y = points[:, 1]
    z = -points[:, 0]
    # Camera centers
    cx1 = c1[2]
    cy1 = c1[1]
    cz1 = -c1[0]
    cx2 = c2[2]
    cy2 = c2[1]
    cz2 = -c2[0]

fig = plt.figure()
fig.set_size_inches(12.5, 8.5)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x, y, z, color="b", alpha=0.8)
ax.scatter(cx1, cy1, cz1, color="r", alpha=0.9)
ax.scatter(cx2, cy2, cz2, color="r", alpha=0.9)


plt.show()

plt.show()