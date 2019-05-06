import sys
from random import Random
from typing import Tuple

import numpy as np

from src.Map import Map
from src.Particle import Particle
from src.detect_resources import detect_resource
from src.detect_water import detect_water


def generate(map: Map, starting_base : Tuple[int, int], resource_range, random):
	"""
	Take a point on earth and returns its position plus the number of resources within the boundary box.
	:param map:
	:param square_range: dimensions of the boundary box
	:param random: random floating point
	:return: coordinates of the particle and the number of resources
	"""
	while True:
		random_point = (random.randint(0, map.map_dim[0]), random.randint(0, map.map_dim[1]))
		# If there is earth under the chosen point break outside, else generate another point
		if map.water_map[random_point[0]][random_point[1]] == 0:  # 0 means earth in the matrix
			break

	# print("Random point: " + str(r_point))
	return Particle(random_point, 0, starting_base, resource_range)


def evaluator(particle: Particle, map: Map):
	"""
	Compute the value of this particle in this location. The formula is: nearby resources - distance
	:return: 
	"""""
	particle.count_resources(map)
	distance = np.linalg.norm(particle.current_position - particle.starting_base)
	print("Distance: " + str(distance))


image_name = sys.argv[1]

map = Map(image_name)

rand: Random = Random()
rand.seed(1)

x = generate(map, resource_range=30, random=rand, starting_base=(0, 0)) # ToDo mettere starting base casuale?
evaluator(x, map)
print("x: " + str(x))
