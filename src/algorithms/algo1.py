import logging
import math
import sys
from random import Random
from typing import Tuple

import coloredlogs as coloredlogs
import numpy as np

from src.algorithms.algorithm import Algorithm
from src.data_structures.Map import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


class Algo1(Algorithm):

    def generate(self, map: Map, starting_base: Tuple[int, int], resource_half_square, random) -> Particle:
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
        return Particle((443, 533), np.zeros(2), starting_base, resource_half_square)
        # return Particle(random_point, 0, starting_base, resource_range)

    def evaluator(self, particle: Particle) -> float:
        """
        Compute the value of this particle in this location. The formula is: nearby resources - distance
        :return: the score of given particle 
        """""
        if self.memo is None:
            logger.debug("The memoization should have been computed")
        else:
            return self.memo[particle.current_position]

    def compute_score(self, particle: Particle, map: Map) -> float:
        res_count = particle.count_resources(map)
        distance = np.linalg.norm(particle.current_position - particle.starting_base)
        # logging.debug("res_count: " + str(res_count))
        # logging.debug("Distance: " + str(distance))

        square_area = (particle.resource_half_square * 2) ** 2
        normalization_factor = math.atan(square_area / map.map_dim[0])
        # logging.debug("math.tan(normalization_factor): " + str(math.tan(normalization_factor)))
        return distance * math.tan(normalization_factor) - res_count

    def __init__(self, map: Map) -> None:
        super().__init__(map)
