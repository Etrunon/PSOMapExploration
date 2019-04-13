from numpy.core.multiarray import ndarray
from matplotlib.figure import Figure
from skimage import io
from skimage.feature import blob_log
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

# This script controls the detection of all different resources: coal, iron, copper and uranium
# Foreach of these create a new image with everything black and only the relevant resource spots white (to ease the job
#        of blob_detection). Then it runs blob_detection and save the result on a new matrix

resources = {
	"coal": [0, 0, 0],
	"iron": [104, 132, 146]
}


def highlight_resource(image, resName):
	result = numpy.zeros((row, col))
	searching = resources[resName]
	# Scan the image looking for the color
	for i in range(0, row):
		for j in range(0, col):
			pixel = img[i][j]
			if pixel[0] == searching[0] and  pixel[1] == searching[1] and  pixel[2] == searching[2]:
				result[i][j] = 1.0
				result[i][j] = 1.0
				result[i][j] = 1.0

	imgPlot = plt.imshow(result)
	plt.show()

	return result


# Load the image
image_name = sys.argv[1]
img = mpimg.imread("data/examples/" + image_name)
row = len(img)
col = len(img[0])

result = highlight_resource(img, "iron")
print("Finito il preprocessing")

blobs_log = blob_log(result, min_sigma=0.1, num_sigma=10, threshold=.05)
print("Finita la ricerca dei blob")

print("X" * 100)
for blob in blobs_log:
	y, x, radius = blob

	if radius > 1:
		print("x: " + str(x) + " y: " + str(y) + " radius: " + str(radius))

# colors = ['yellow', 'lime', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
# sequence = zip(blobs_list, colors, titles)
#
# fig: Figure
# axes: ndarray
# fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()

# for idx, (blobs, color, title) in enumerate(sequence):
# 	ax[idx].set_title(title)
# 	ax[idx].imshow(result, interpolation='nearest')
# 	for blob in blobs:
# 		y, image_path, r = blob
# 		c = plt.Circle((image_path, y), r, color=color, linewidth=0.1, fill=False)
# 		ax[idx].add_patch(c)
# 	ax[idx].set_axis_off()
#
# plt.tight_layout()
# plt.show()
