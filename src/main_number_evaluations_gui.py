import csv
import logging

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import Circle

from src.algorithms.algo1 import Algo1
from src.configuration import IMAGE_NAME, CSV_FILE
from src.data_structures.Map import Map

IMAGE_PATH = 'data/examples/{}'.format(IMAGE_NAME)

# Set the theme used by matplotlib
plt.style.use("seaborn-bright")

# Set the backend used by matplotlib
matplotlib.use("Qt5Agg")

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    positions = []

    with open(CSV_FILE, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for line in reader:
            position = int(line['position_x']), int(line['position_y']), float(line['fitness'])

            positions.append(position)
            starting_city = int(line['city_position_x']), int(line['city_position_y'])

    # Create a figure, because inspyred library already creates one
    figure: Figure = matplotlib.pyplot.figure()

    # Save a reference to the axes object
    ax: Axes = figure.add_subplot(1, 1, 1)

    image = imread(IMAGE_PATH)

    x, y, fitness = zip(*positions)

    # plot = ax.scatter(x, y, c=fitness, zorder=1, cmap='Spectral')
    plot = ax.scatter(x, y, c='blue', zorder=1)

    # Plot the starting position as a red circle
    starting_city_circle = Circle(starting_city, 10, facecolor="red", alpha=1)
    ax.add_patch(starting_city_circle)

    # Use the map as background
    ax.imshow(image, zorder=0)

    ########
    # calculate the fitness on all the map and take the maximum value

    logger.info("Calculating the best fitness point manually")
    world_map = Map(IMAGE_NAME)
    algorithm = Algo1(world_map, 137, 100, False)
    landscape = algorithm.get_fitness_landscape()

    minimum = np.amin(landscape)
    minimum_positions = np.where(landscape == minimum)
    minimum_position = minimum_positions[0][0], minimum_positions[1][0]

    logger.info("Best fitness is %s at %d %d", minimum, minimum_position[0], minimum_position[1])

    # Plot the best position as a white circle
    minimum_position_circle = Circle(minimum_position, facecolor="purple", alpha=0.6)
    ax.add_patch(minimum_position_circle)

    ax.annotate(
        'best position',
        xy=minimum_position,
        xytext=(position + 40 for position in minimum_position),
        arrowprops=dict(arrowstyle='->', color="white"),
        color="white"
    )

    plt.show(block=True)
