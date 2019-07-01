import logging
import sys
from random import Random

import coloredlogs
import inspyred
import matplotlib.pyplot
import numpy as np
from PIL import Image
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src.algorithms.algo1 import Algo1
from src.configuration import RESOURCE_RANGE, CITY_POSITION, POPULATION_SIZE, COGNITIVE_RATE, INERTIA_RATE, \
    SOCIAL_RATE
from src.custom_pso import evaluate_particle, custom_terminator, custom_variator, \
    custom_observer, CustomPSO
from src.data_structures.Map import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)

# Set the theme used by matplotlib
matplotlib.pyplot.style.use("seaborn-bright")
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

    algorithm.plot_fitness_landscape()
