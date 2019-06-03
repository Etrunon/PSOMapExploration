import logging
import math
import sys
from multiprocessing.pool import Pool
from random import Random
from typing import Tuple

import coloredlogs as coloredlogs
import numpy as np

from src.data_structures.Map import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


def generate(map: Map, starting_base: Tuple[int, int], resource_half_square, random):
    """
    Take a point on earth and returns its position plus the number of resources within the boundary box.
    :param resource_half_square: range in which the particle will be looking for resources
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
    return Particle((443, 533), 0, starting_base, resource_half_square)
    # return Particle(random_point, 0, starting_base, resource_range)


def evaluator(particle: Particle, map: Map):
    """
    Compute the value of this particle in this location. The formula is: nearby resources - distance
    :return: the score of given particle 
    """""
    res_count = particle.count_resources(map)
    distance = np.linalg.norm(particle.current_position - particle.starting_base)
    logging.debug("res_count: " + str(res_count))
    logging.debug("Distance: " + str(distance))

    square_area = (particle.resource_half_square * 2) ** 2
    normalization_factor = math.atan(square_area / map.map_dim[0])
    logging.debug("math.tan(normalization_factor): " + str(math.tan(normalization_factor)))
    return distance * math.tan(normalization_factor) - res_count


STARTING_BASE = (0, 0)
RESOURCE_RADIUS = 25


def count_resources(x, y) -> Tuple[int, int, int]:
    if x % 100 == 0 and y % 600 == 0:
        logger.debug("Doing %d %d", x, y)
    particle = Particle((x, y), 0, (0, 0), resource_range=RESOURCE_RADIUS)
    return particle.count_resources(map), x, y


def result_callback(args):
    resources, x, y = args
    result[x][y] = resources


def error_callback(exception):
    print(exception)


def calculate_resources() -> np.array:
    dimension_x, dimension_y = map.map_dim

    global result
    result = [[None for x in range(dimension_y)] for x in range(dimension_x)]

    with Pool() as pool:
        for i in range(0, dimension_x):
            for j in range(0, dimension_y):
                async = pool.apply_async(count_resources,
                                         (i, j),
                                         callback=result_callback,
                                         error_callback=error_callback)

        pool.close()
        pool.join()

    print(type(result))
    return np.array(result, dtype=np.uint)


result = None

if __name__ == "__main__":
    coloredlogs.install(level='DEBUG', style='{', fmt='{name:15s} {levelname} {message}')

    image_name = sys.argv[1]

    map = Map(image_name)

    rand: Random = Random()
    rand.seed(1)

    cached_matrix_path = 'data/cached_matrices/' + image_name.replace('.png', '') + '_resource_count.npy'
    try:
        resources = np.load(cached_matrix_path)
        logger.info("Using cached matrix")
    except IOError:

        logger.info('Resource matrix file does not exist or cannot be read.')
        resources = calculate_resources()

        np.save(cached_matrix_path, resources)
        logger.info('Resource processing completed!')
