import logging
from multiprocessing.pool import Pool
from typing import Tuple

import numpy as np

from src.configuration import RESOURCE_RANGE
from src.data_structures.Map import Map, map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)
result = None


def count_resources(x: int, y: int) -> Tuple[int, int, int]:
    # Print a log every x and y cycles
    if x % 100 == 0 and y % 600 == 0:
        logger.info("Calculating on coordinates x = %d, y = %d...", x, y)
    # Perform the computation
    particle = Particle((x, y), 0, (0, 0), resource_range=RESOURCE_RANGE)
    return particle.count_resources(map), x, y


def result_callback(args):
    resources, x, y = args
    result[x][y] = resources


def calculate_resources(map: Map) -> np.array:
    dimension_x, dimension_y = map.map_dim

    global result
    # Initialize an empty list of list
    result = [[None for x in range(dimension_y)] for x in range(dimension_x)]

    # Start a pool to parallelize the computation
    with Pool() as pool:
        for i in range(0, dimension_x):
            for j in range(0, dimension_y):
                async = pool.apply_async(count_resources,
                                         (i, j),
                                         callback=result_callback)

        pool.close()
        pool.join()

    return np.array(result, dtype=np.uint)
