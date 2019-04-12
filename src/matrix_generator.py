import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load  image
image_path = sys.argv[1]
image = Image.open(image_path)
data = np.asarray(image, dtype="int32")

# Create matrix for the result
rows = data.shape[0]
columns = data.shape[1]
result = np.zeros((rows, columns))

# Scan image looking for the colors
for r in range(0, rows):
	for c in range(0, columns):
		if (data[r][c][0] == 38 and data[r][c][1] == 64 and data[r][c][2] == 73) or (
			data[r][c][0] == 51 and data[r][c][1] == 83 and data[r][c][2] == 95):
			result[r][c] = 1

# Plot result
plt.imshow(result)
plt.show()
