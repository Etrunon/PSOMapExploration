from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import sys

image_path = sys.argv[1]
image = io.imread(image_path)

rows = image.shape[0]
columns = image.shape[1]

mask = my_array = np.zeros([rows, columns])

for r in range(rows):
	for c in range(columns):
		pixel = image[r][c]
		if (pixel[0] == 38 and pixel[1] == 64 and pixel[2] == 73) or (
			pixel[0] == 51 and pixel[1] == 83 and pixel[2] == 95):
			mask[r][c] = 0
		else:
			mask[r][c] = 1

plt.imshow(mask)
plt.show()
