import logging
import math
from _random import Random

import numpy as np

from src.algorithms.algorithm import Algorithm
from src.configuration import CITY_POSITION
from src.data_structures import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)

id = -1


class Algo1(Algorithm):

    def generate_particle(self, random: Random, args) -> Particle:
        """

        Take a point on earth and returns its position plus the number of resources within the boundary box.

        Parameters:
            random: random generator
        Returns:
            The newly created particle
        """
        global id

        while True:
            random_point = (random.randint(self.resource_range, self.world_map.map_dim[0] - self.resource_range),
                            random.randint(self.resource_range, self.world_map.map_dim[1] - self.resource_range))

            # If there is earth under the chosen point break outside, else generate another point
            if self.world_map.water_map[random_point[0]][random_point[1]] == 0:  # 0 means earth in the matrix
                break

        # ToDo inizializzare bene il vettore, perchè fatto così arriva fino a ben oltre la MAXIMUM_VELOCITY
        # // is integer division
        velocity = (random.randint(1, self.maximum_velocity // 2), random.randint(1, self.maximum_velocity // 2))

        id += 1

        return Particle(starting_position=random_point,
                        velocity=np.array(velocity, np.uintc),
                        resource_range=self.resource_range,
                        starting_base=CITY_POSITION,
                        id=id,
                        world_map=self.world_map,
                        save_history=self.save_history)

    def evaluate_particle(self, particle: Particle) -> float:
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

        # If we have the resource count matrix, simply read from it
        # res_count = self.resource_count_matrix[particle.current_position[0], particle.current_position[1]]

        # If we don't have the resource count matrix, calculate the resources
        res_count = particle.count_resources()

        distance = np.linalg.norm(particle.current_position - particle.starting_base)

        # DEBUG
        # logging.debug("res_count: " + str(res_count))
        # logging.debug("Distance: " + str(distance))

        square_area = (self.resource_range * 2) ** 2
        normalization_factor = math.atan(self.world_map.map_dim[0] / square_area)

        # logging.debug("math.tan(normalization_factor): " + str(math.tan(normalization_factor)))
        return distance * math.tan(normalization_factor) - res_count

    def __init__(self, world_map: Map, maximum_velocity: int, resource_range: int, save_history: bool) -> None:
        super().__init__(world_map, resource_range)
        self.maximum_velocity = maximum_velocity
        self.save_history = save_history

    def get_fitness_landscape(self) -> np.ndarray:
        """
        Returns:
            The fitness landscape as two dimensional array
        """
        landscape = np.zeros(self.world_map.map_dim)

        for x in range(self.resource_range, self.world_map.map_dim[0] - self.resource_range):
            for y in range(self.resource_range, self.world_map.map_dim[1] - self.resource_range):
                sensor = Particle((x, y), np.array((1, 1), np.uintc), CITY_POSITION, self.resource_range, None, 0)
                landscape[x][y] = self.compute_score(sensor)

        return landscape
