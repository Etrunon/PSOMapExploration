import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.core.multiarray import ndarray
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import blob_log
import sys

image_path = sys.argv[1]
image = io.imread(image_path)
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.05)
print(blobs_log)

# Compute radii in the 3rd column.
# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

# blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
# blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

# blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

# blobs_list = [blobs_log, blobs_dog, blobs_doh]
blobs_list = [blobs_log]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
		  'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig: Figure
axes: ndarray
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
	ax[idx].set_title(title)
	ax[idx].imshow(image, interpolation='nearest')
	for blob in blobs:
		y, image_path, r = blob
		c = plt.Circle((image_path, y), r, color=color, linewidth=2, fill=False)
		ax[idx].add_patch(c)
	ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
