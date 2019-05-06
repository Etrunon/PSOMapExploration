import math
import sys
from random import Random
from typing import Tuple
from inspyred import benchmarks

import numpy as np

from src.Map import Map
from src.Particle import Particle


def generate(map: Map, starting_base: Tuple[int, int], resource_radius, random):
    """
    Take a point on earth and returns its position plus the number of resources within the boundary box.
    :param resource_radius: range in which the particle will be looking for resources
    :param starting_base: base of operations
    :param map:
    :param random: random generator
    :return: the new particle
    """
    while True:
        random_point = (random.randint(0, map.map_dim[0]), random.randint(0, map.map_dim[1]))
        # If there is earth under the chosen point break outside, else generate another point
        if map.water_map[random_point[0]][random_point[1]] == 0:  # 0 means earth in the matrix
            break

    # print("Random point: " + str(r_point))
    return Particle((443, 533), 0, starting_base, resource_radius)
    # return Particle(random_point, 0, starting_base, resource_range)


def evaluator(particle: Particle, map: Map):
    """
    Compute the value of this particle in this location. The formula is: nearby resources - distance
    :return: the score of given particle 
    """""
    res_count = particle.count_resources(map)
    distance = np.linalg.norm(particle.current_position - particle.starting_base)
    print("res_count: " + str(res_count))
    print("Distance: " + str(distance))

    square_area = (particle.resource_radius*2)**2
    normalization_factor = math.atan(square_area / map.map_dim[0])
    print("math.tan(normalization_factor): " + str(math.tan(normalization_factor)))
    return distance * math.tan(normalization_factor) - res_count


image_name = sys.argv[1]

map = Map(image_name)

rand: Random = Random()
rand.seed(1)

x = generate(map, resource_radius=25, random=rand, starting_base=(0, 0))  # ToDo mettere starting base casuale?
score = evaluator(x, map)
print("x: " + str(x))
print("score x: " + str(score))

print(str())
