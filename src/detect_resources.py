from skimage import io
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

# This script controls the detection of all different resources: coal, iron, copper and uranium
# Foreach of these create a new image with everything black and only the relevant resource spots white (to ease the job
#        of blob_detection). Then it runs blob_detection and save the result on a new matrix

# Load the image
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
