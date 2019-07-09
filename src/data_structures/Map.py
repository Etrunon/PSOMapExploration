import string

import numpy as np

from src.configuration import IMAGE_NAME, RESOURCE_RANGE
from src.preprocess_utilities.detect_resources import detect_resource
from src.preprocess_utilities.detect_water import detect_water


class Map:
    """
    Represents a world map

    """

    resource_map = None
    water_map = None
    map_dim = None

    best_position: np.ndarray = None  #: Array containing the x,y coordinates of the global best
    best_fitness: float = 0  #: Value of the fitness in the best position

    def __init__(self, image_name: string):
        self.resource_map = detect_resource(image_name)
        self.water_map = detect_water(image_name)
        self.map_dim = self.resource_map.shape

    def is_inside_map(self, position: np.ndarray) -> bool:
        """
        Subtract the resource range, so that each point having true as output, has a relevant square around it.
        If it did not subtract the range from the then it would be possible to compute the value for point 1,1 even
        though it would have only 51 x 51 points around, instead of 100x100.

        Returns:
            bool: true if the position is inside the map, false otherwise
        """
        x = position[0]
        y = position[1]

        if RESOURCE_RANGE < x < self.map_dim[0] - RESOURCE_RANGE:
            if RESOURCE_RANGE < y < self.map_dim[1] - RESOURCE_RANGE:
                return True

        return False


world_map: Map = Map(image_name=IMAGE_NAME)
