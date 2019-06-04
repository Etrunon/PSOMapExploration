import string

import numpy as np

from src.configuration import IMAGE_NAME
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

    def out_of_map(self, position: np.ndarray) -> bool:
        """

        Returns:
            bool: true if the position is inside the map, false otherwise
        """
        for coordinate in position:
            if coordinate > self.map_dim[0] or coordinate < 0:
                return True

        return False


map: Map = Map(image_name=IMAGE_NAME)
