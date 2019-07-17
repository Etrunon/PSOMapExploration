import logging
import sys
from typing import Tuple

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
    resource_range = None

    movements: np.ndarray = np.zeros([0, 2])  #: Array containing all the positions
    velocities: np.ndarray = np.zeros([0, 2])  #: Array containing all the velocities

    def __init__(self, starting_position: Tuple[int, int], velocity: np.ndarray, starting_base: Tuple[int, int],
                 resource_range: int, world_map: Map, id: int, save_history: bool):
        self.world_map = world_map
        self.current_position = np.array([starting_position[0], starting_position[1]])
        self.velocity = velocity
        self.best_position = self.current_position
        self.best_fitness = sys.float_info.max  # Initialize best fitness to an absurd high value

        self.starting_base = np.array([starting_base[0], starting_base[1]])
        self.resource_range = resource_range
        self.id = id
        self.save_history = save_history

    def count_resources(self):
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
            self.current_position[0] - self.resource_range, self.current_position[1] - self.resource_range)
        v2 = (
            self.current_position[0] + self.resource_range, self.current_position[1] - self.resource_range)
        v3 = (
            self.current_position[0] + self.resource_range, self.current_position[1] + self.resource_range)
        v4 = (
            self.current_position[0] - self.resource_range, self.current_position[1] + self.resource_range)

        true_v1 = (max(v1[0], 0), max(v1[1], 0))
        true_v2 = (min(v2[0], self.world_map.map_dim[0]), max(v2[1], 0))
        true_v3 = (min(v3[0], self.world_map.map_dim[0]), min(v3[1], self.world_map.map_dim[1]))
        true_v4 = (max(v4[0], 0), min(v4[1], self.world_map.map_dim[1]))

        # Now that we have the box, let's count how many resources are inside
        sub_matrix = self.world_map.resource_map[true_v1[0]:true_v2[0], true_v1[1]:true_v4[1]]

        return np.sum(sub_matrix)

    def move_to(self, new_position: np.ndarray):
        """
        Set the particle to the given position, saving its old one inside the movements array

        Args:
             new_position: Array with x and y coordinates
        """

        if self.save_history:
            self.movements = np.append(self.movements, [self.current_position], axis=0)
        self.current_position = new_position

    def set_velocity(self, new_velocity: np.ndarray):  # TODO: needed?
        """
        Set the particle's velocity, saving its old one inside the velocities array

        Args:
             new_velocity: Array with x and y velocity
        """
        if self.save_history:
            self.velocities = np.append(self.velocities, [self.velocity], axis=0)
        self.velocity = new_velocity

    def __str__(self) -> str:
        return "\t\t\tcurrent_position: " + str(self.current_position) + " \n" + \
               "\t\t\tvelocity: " + str(self.velocity) + " \n" + \
               "\t\t\tlocal_best: " + str(self.best_position) + " \n" + \
               "\t\t\tstarting_base: " + str(self.starting_base) + " \n" + \
               "\t\t\tresource_range: " + str(self.resource_range) + " \n"
