import string
import sys

import numpy as np

from src.preprocess_utilities.detect_resources import detect_resources


class Map:
    """
    Represents a world map

    """

    resource_map = None
    water_map = None
    map_dim = None

    best_position: np.ndarray = None  #: Array containing the x,y coordinates of the global best
    # Initialize best fitness to an absurd high value
    best_fitness: float = sys.float_info.max  #: Value of the fitness in the best position

    def __init__(self, image_name: string):
        self.resource_map = detect_resources(image_name)
        self.water_map = detect_resources(image_name, water=True)
        self.map_dim = self.resource_map.shape

    def is_inside_map(self, position: np.ndarray, resource_range: int) -> bool:
        """
        Subtract the resource range, so that each point having true as output, has a relevant square around it.
        If it did not subtract the range from the then it would be possible to compute the value for point 1,1 even
        though it would have only 51 x 51 points around, instead of 100x100.

        Returns:
            bool: true if the position is inside the map, false otherwise
        """
        x = position[0]
        y = position[1]

        if resource_range < x < self.map_dim[0] - resource_range:
            if resource_range < y < self.map_dim[1] - resource_range:
                return True

        return False


world_map = None
