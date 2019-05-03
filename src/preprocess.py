from PIL import Image

import sys

import numpy as np
import matplotlib.pyplot as plt


# Load the image
from src.detect_resources import detect_resource
from src.detect_water import detect_water

image_name = sys.argv[1]
img = Image.open("data/examples/" + image_name)
img_array = np.asarray(img, dtype="int32")

processed = np.zeros((2, img_array.shape[0], img_array.shape[1]))
processed[0] = detect_resource(img_array)
processed[1] = detect_water(img_array)

# Plot result
plt.imshow(img_array)
plt.show()

plt.imshow(processed[0])
plt.show()

plt.imshow(processed[1])
plt.show()
