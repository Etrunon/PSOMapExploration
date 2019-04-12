from numpy.core.multiarray import ndarray
from matplotlib.figure import Figure
from skimage import io
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

# This script controls the detection of all different resources: coal, iron, copper and uranium
# Foreach of these create a new image with everything black and only the relevant resource spots white (to ease the job
#        of blob_detection). Then it runs blob_detection and save the result on a new matrix

# Load the image
from skimage.feature import blob_log

image_path = sys.argv[1]
img = mpimg.imread(image_path)

# Create matrix for the result
row = len(img)
col = len(img[0])
result = numpy.zeros((row, col))

# Scan the image looking for the color
for i in range(0, row):
	for j in range(0, col):
		if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
			result[i][j] = 1.0
			result[i][j] = 1.0
			result[i][j] = 1.0

# Plot result
imgPlot = plt.imshow(result)
plt.show()

print("Finito il preprocessing")

blobs_log = blob_log(result, min_sigma=3, num_sigma=10, threshold=.05)
print(blobs_log)

blobs_list = [blobs_log]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig: Figure
axes: ndarray
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
	ax[idx].set_title(title)
	ax[idx].imshow(result, interpolation='nearest')
	for blob in blobs:
		y, image_path, r = blob
		c = plt.Circle((image_path, y), r, color=color, linewidth=2, fill=False)
		ax[idx].add_patch(c)
	ax[idx].set_axis_off()

plt.tight_layout()
plt.show()


