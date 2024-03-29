import logging
from random import Random

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src.algorithms.algo1 import Algo1
from src.configuration import IMAGE_NAME, MAXIMUM_VELOCITY, RESOURCE_RANGE, CITY_POSITION
from src.data_structures.Map import Map

#########################
# This file plots the fitness landscape as an overlay on top of the map
##########################

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
    rand.seed(1)

    world_map = Map(IMAGE_NAME)
    algorithm = Algo1(world_map, MAXIMUM_VELOCITY, RESOURCE_RANGE, False)

    logger.info("Calculating fitness on the whole map")
    landscape = algorithm.get_fitness_landscape()

    figure: Figure = matplotlib.pyplot.figure()

    # Save a reference to the axes object
    ax: Axes = figure.add_subplot(1, 1, 1)

    ax.imshow(Image.open('data/examples/{}'.format(IMAGE_NAME)), interpolation='nearest')

    # Add heatmap
    mat = ax.matshow(np.transpose(landscape), alpha=0.5)

    # Add colorbar explaining the colours used in the heatmap
    figure.colorbar(mat)

    # Plot the starting position as a red circle
    start = Circle(CITY_POSITION, 10, facecolor="red", alpha=1)

    ax.add_patch(start)

    # Print the best fitness location and value
    minimum = np.amin(landscape)
    minimum_positions = np.where(landscape == minimum)
    minimum_position = minimum_positions[0][0], minimum_positions[1][0]

    logger.info("Best fitness is %s at %d %d", minimum, minimum_position[0], minimum_position[1])
    # Plot the best position as a white circle
    minimum_position_circle = Circle(minimum_position, facecolor="purple", alpha=0.6)
    ax.add_patch(minimum_position_circle)

    ax.annotate(
        'best position %d'.format(minimum),
        xy=minimum_position,
        xytext=(position + 40 for position in minimum_position),
        arrowprops=dict(arrowstyle='->', color="white"),
        color="white"
    )

    matplotlib.pyplot.show(block=True)
