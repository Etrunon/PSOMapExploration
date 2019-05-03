import sys
from random import Random

import numpy as np
from PIL import Image

from src.detect_resources import detect_resource
from src.detect_water import detect_water


def generator(square_range, rand):
	while True:
		r_point = (rand.randint(0, map_dim[0]), rand.randint(0, map_dim[1]))
		# If there is no water under the chosen point break
		if water_map[r_point[0]][r_point[1]] == 0:
			break

	# print("Random point: " + str(r_point))

	# Now we compute the value of all resources around.
	# We do this computing first the actual bounding box around the chosen point, making sure not to get outside the
	#    limits of the matrix. (in the case the chosen point is too close to the border)
	# The idea is to check if each vertex of the bounding box is in a legal position and if not replace it with the
	#    border.
	#
	# v1                   v2
	#      true_v1------true_v2--------
	#      |
	#      |      cp
	#      |
	# v4   true_v4         v3

	v1 = (r_point[0] - square_range, r_point[1] - square_range)
	v2 = (r_point[0] + square_range, r_point[1] - square_range)
	v3 = (r_point[0] + square_range, r_point[1] + square_range)
	v4 = (r_point[0] - square_range, r_point[1] + square_range)

	true_v1 = (max(v1[0], 0), max(v1[1], 0))
	true_v2 = (min(v2[0], map_dim[0]), max(v2[1], 0))
	true_v3 = (min(v3[0], map_dim[0]), min(v3[1], map_dim[1]))
	true_v4 = (max(v4[0], 0), min(v4[1], map_dim[1]))

	# print("true_v1: " + str(true_v1))
	# print("true_v2: " + str(true_v2))
	# print("true_v3: " + str(true_v3))
	# print("true_v4: " + str(true_v4))

	# Now that we have to box, lets count how many resources are inside
	res_found = 0
	for i in range(true_v1[0], true_v2[0]):
		for j in range(true_v1[1], true_v4[1]):
			if resource_map[i][j] != 0:
				res_found = res_found + 1

	# print("res_found " + str(res_found))

	return r_point[0], r_point[1], res_found


image_name = sys.argv[1]
img = Image.open("data/examples/" + image_name)
img_array = np.asarray(img, dtype="int32")

# Returns a 2 x height x width matrix.
# The first contains resources while the second contains water
resource_map = detect_resource(img_array)
water_map = detect_water(img_array)

map_dim = resource_map.shape

rand: Random = Random()
rand.seed(1)

generator(square_range=30, rand=rand)
