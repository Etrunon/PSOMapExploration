import logging
import sys
from random import Random

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt

from src.algorithms.algo1 import Algo1
from src.data_structures.Map import Map

logger = logging.getLogger(__name__)

# Set the theme used by matplotlib
plt.style.use("seaborn-bright")
# Set the backend used by matplotlib
matplotlib.use("Qt5Agg")

if __name__ == "__main__":
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    # Initialize the random seed
    rand = Random()
    rand.seed(1)  # TODO: set to 1 for debug purposes, remove once ready to take off!

    image_name = sys.argv[1]
    world_map = Map(image_name)
    algorithm = Algo1(world_map)

    logger.info("Calculating fitness")
    landscape = algorithm.get_fitness_landscape()

    plt.matshow(landscape)
    plt.show()
