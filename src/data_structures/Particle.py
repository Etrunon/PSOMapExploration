import logging
import time

import numpy as np

from src.data_structures.Map import Map

logger = logging.getLogger(__name__)


class Particle:
    """
    A single individual in a swarm
    """

    current_position = (None, None)
    velocity: np.ndarray = None
    # Location of the "city" which we're trying to optimize
    starting_base = (None, None)

    """Radius in which this particle will look for resource. It is a square (not a circle) and this number represents
    half side (i.e. if this number is 10 the square has side 20 and area 400)"""
    resource_half_square = None

    best_fitness: float = 0
    best_position: np.ndarray = None

    movements: np.ndarray = np.zeros([0, 2])  #: Array containing all the positions
    velocities: np.ndarray = np.zeros([0, 2])  #: Array containing all the velocities

    def __init__(self, starting_position, velocity: np.ndarray, starting_base, resource_range):
        self.current_position = np.array([starting_position[0], starting_position[1]])
        self.velocity = velocity
        self.local_best = self.current_position

        self.starting_base = np.array([starting_base[0], starting_base[1]])
        self.resource_half_square = resource_range
        self.resource_radius = resource_range

    def count_resources(self, map: Map):
        """
        Compute the value of all resources around the square.
        The computation finds the actual bounding box around the chosen point, making sure not to get outside the
        limits of the matrix (in the case the chosen point is too close to the border).
        The idea is to check if each vertex of the bounding box is in a legal position and if not replace it with the border.
        As a note, v3 is actually not needed.

        Returns:
            The number of resources
        """
        # v1                   v2
        #      true_v1------true_v2--------
        #      |
        #      |      cp
        #      |
        # v4   true_v4         v3

        v1 = (
            self.current_position[0] - self.resource_half_square, self.current_position[1] - self.resource_half_square)
        v2 = (
            self.current_position[0] + self.resource_half_square, self.current_position[1] - self.resource_half_square)
        v3 = (
            self.current_position[0] + self.resource_half_square, self.current_position[1] + self.resource_half_square)
        v4 = (
            self.current_position[0] - self.resource_half_square, self.current_position[1] + self.resource_half_square)

        true_v1 = (max(v1[0], 0), max(v1[1], 0))
        true_v2 = (min(v2[0], map.map_dim[0]), max(v2[1], 0))
        true_v3 = (min(v3[0], map.map_dim[0]), min(v3[1], map.map_dim[1]))
        true_v4 = (max(v4[0], 0), min(v4[1], map.map_dim[1]))

        # Now that we have the box, let's count how many resources are inside
        res_found = 0
        logger.debug("square range: {} {}, {} {}".format(true_v1[0], true_v2[0], true_v3[0], true_v4[0]))

        start = time.time()

        for i in range(true_v1[0], true_v2[0]):
            for j in range(true_v1[1], true_v4[1]):
                if map.resource_map[i][j] != 0:
                    res_found = res_found + 1

        # Test the time required by the algorithm
        # logger.debug("Time required: %f", time.time() - start)

        # Test the time required by numpy
        # start = time.time()
        # slice = map.resource_map[true_v1[0]:true_v2[0], true_v1[1]:true_v4[1]]
        #
        # count = np.count_nonzero(slice)
        #
        # logger.debug("Time required numpy: %f", time.time() - start)

        return res_found

    def move_to(self, new_position: np.ndarray):
        """
        Set the particle to the given position, saving its old one inside the movements array

        Args:
             new_position: Array with x and y coordinates
        """

        self.movements = np.append(self.movements, [self.current_position], axis=0)
        self.current_position = new_position

    def set_velocity(self, new_velocity: np.ndarray):  # TODO: needed?
        """
        Set the particle's velocity, saving its old one inside the velocities array

        Args:
             new_velocity: Array with x and y velocity
        """
        self.velocities = np.append(self.velocities, [self.velocity], axis=0)
        self.velocity = new_velocity

    def __str__(self) -> str:
        return "current_position: " + str(self.current_position) + " \n" + \
               "\tvelocity: " + str(self.velocity) + " \n" + \
               "\tlocal_best: " + str(self.local_best) + " \n" + \
               "\tstarting_base: " + str(self.starting_base) + " \n" + \
               "\tresource_range : " + str(self.resource_half_square) + " \n"
