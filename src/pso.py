import sys
from random import Random

from src.detect_resources import detect_resource
from src.detect_water import detect_water


def count_resources(r_point, square_range):
	"""
	Compute the value of all resources around the square.
	The computation finds the actual bounding box around the chosen point, making sure not to get outside the
	limits of the matrix (in the case the chosen point is too close to the border).
	The idea is to check if each vertex of the bounding box is in a legal position and if not replace it with the border.
	As a note, v3 is actually not needed.
	:param r_point: random position on the map
	:param square_range: dimensions of the boundary box
	:return: the number of resources
	"""

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

	# Now that we have the box, let's count how many resources are inside
	res_found = 0
	for i in range(true_v1[0], true_v2[0]):
		for j in range(true_v1[1], true_v4[1]):
			if resource_map[i][j] != 0:
				res_found = res_found + 1

	# print("res_found " + str(res_found))
	return res_found


def particle_generator(square_range, random):
	"""
	Take a point on earth and returns its position plus the number of resources within the boundary box.
	:param square_range: dimensions of the boundary box
	:param random: random floating point
	:return: coordinates of the particle and the number of resources
	"""
	while True:
		random_point = (random.randint(0, map_dim[0]), random.randint(0, map_dim[1]))
		# If there is water under the chosen point break
		if water_map[random_point[0]][random_point[1]] == 1:
			break

	# print("Random point: " + str(r_point))
	return random_point[0], random_point[1], count_resources(random_point, square_range)


image_name = sys.argv[1]

resource_map = detect_resource(image_name)
water_map = detect_water(image_name)

map_dim = resource_map.shape

rand: Random = Random()
rand.seed(1)

x = particle_generator(square_range=3, random=rand)
print("x: " + str(x))
