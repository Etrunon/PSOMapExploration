import logging
import math
from _random import Random

import numpy as np

from src.algorithms.algorithm import Algorithm
from src.configuration import RESOURCE_RANGE, MAXIMUM_VELOCITY, CITY_POSITION
from src.data_structures.Map import world_map as world_map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


class Algo1(Algorithm):

    def generate_particle(self, random: Random, args) -> Particle:
        """
        Take a point on earth and returns its position plus the number of resources within the boundary box.
        :param resource_half_square: range in which the particle will be looking for resources
        :param starting_base: base of operations
        :param map:
        :param random: random generator
        :return: the new particle
        """
        while True:
            random_point = (random.randint(RESOURCE_RANGE, world_map.map_dim[0] - RESOURCE_RANGE),
                            random.randint(RESOURCE_RANGE, world_map.map_dim[1] - RESOURCE_RANGE))
            # If there is earth under the chosen point break outside, else generate another point
            if world_map.water_map[random_point[0]][random_point[1]] == 0:  # 0 means earth in the matrix
                break

        # ToDo inizializzare bene il vettore, perchè fatto così arriva fino a ben oltre la MAXIMUM_VELOCITY
        velocity = (random.randint(1, MAXIMUM_VELOCITY / 2), random.randint(1, MAXIMUM_VELOCITY / 2))

        return Particle(starting_position=random_point,
                        velocity=np.array(velocity, np.uintc),
                        resource_range=RESOURCE_RANGE,
                        starting_base=CITY_POSITION)

    def evaluator(self, particle: Particle) -> float:
        """
        Compute the value of this particle in this location. The formula is: nearby resources - distance
        :return: the score of given particle 
        """""
        # if self.memo is None:
        #     logger.debug("The memoization should have been computed")
        # else:
        #     return self.memo[particle.current_position]
        # TODO: enable memoization once works
        return self.compute_score(particle)

    def compute_score(self, particle: Particle) -> float:

        res_count = self.resource_count_matrix[particle.current_position[0], particle.current_position[1]]
        distance = np.linalg.norm(particle.current_position - particle.starting_base)
        # logging.debug("res_count: " + str(res_count))
        # logging.debug("Distance: " + str(distance))

        square_area = (RESOURCE_RANGE * 2) ** 2
        normalization_factor = math.atan(self.resource_count_matrix.shape[0] / square_area)
        # logging.debug("math.tan(normalization_factor): " + str(math.tan(normalization_factor)))
        return distance * math.tan(normalization_factor) - res_count

    def __init__(self) -> None:
        super().__init__()

    def get_fitness_landscape(self) -> np.ndarray:
        """
        Returns:
            The fitness landscape as two dimensional array
        """
        landscape = np.zeros(world_map.map_dim)

        for x in range(RESOURCE_RANGE, world_map.map_dim[0] - RESOURCE_RANGE):
            for y in range(RESOURCE_RANGE, world_map.map_dim[1] - RESOURCE_RANGE):
                sensor = Particle((x, y), np.array((1, 1), np.uintc), CITY_POSITION, RESOURCE_RANGE, "sensor")
                landscape[x][y] = self.compute_score(sensor)

        return landscape
