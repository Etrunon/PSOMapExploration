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
from src.configuration import RESOURCE_RANGE, STARTING_POSITION, POPULATION_SIZE
from src.custom_pso import generate_particle, evaluate_particle, custom_terminator, custom_variator, \
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
    coloredlogs.install(level='DEBUG', style='{', fmt='{name:15s} {levelname} {message}')

    # Initialize the random seed
    rand = Random()
    rand.seed(1)  # TODO: set to 1 for debug purposes, remove once ready to take off!

    image_name = sys.argv[1]
    world_map = Map(image_name)
    algorithm = Algo1(world_map)

    # Create a figure, because inspyred already creates one
    figure: Figure = matplotlib.pyplot.figure(2)

    # Plot a grid in the figure
    pyplot.grid()

    # Use the map as background
    matplotlib.pyplot.imshow(Image.open('data/examples/' + image_name))

    # Save a reference to the axes object
    ax: Axes = figure.add_subplot(111)
    # Force an aspect for the axes
    ax.set_aspect('equal')

    # TODO: try out if this helps
    # ax.use_sticky_edges = False

    # Plot the starting position as a red circle
    start = Circle(STARTING_POSITION, 10, facecolor="red", alpha=1)
    ax.add_patch(start)

    # Use minimal padding inside the figure
    matplotlib.pyplot.tight_layout(pad=0)

    # Instantiate the custom PSO instance
    custom_pso = CustomPSO(rand)

    # Set the map and the algorithm
    custom_pso.set_world_map(world_map)
    custom_pso.set_algorithm(algorithm)

    # Set custom properties for the PSO instance
    custom_pso.terminator = custom_terminator

    # Set the custom variator to move each particle in the swarm
    custom_pso.variator = custom_variator

    # Set the topology to specify how neighbours are found
    custom_pso.topology = inspyred.swarm.topologies.star_topology

    # Observers (custom logger that are notified while the algorithm runs)
    custom_pso.observer = [inspyred.ec.observers.plot_observer, custom_observer]

    # Create a new figure, to be used by inspyred plot_observer
    figure: Figure = matplotlib.pyplot.figure(1)

    # Run the PSO algorithm
    final_population = custom_pso.evolve(generator=generate_particle,
                                         # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                                         evaluator=evaluate_particle,
                                         # mp_evaluator=fitness_evaluator,
                                         pop_size=POPULATION_SIZE,
                                         maximize=False,
                                         bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                         # neighborhood_size=5,
                                         max_evaluations=500,
                                         # statistics_file=stat_file,
                                         # individuals_file=ind_file)
                                         inertia=1,
                                         cognitive_rate=1,
                                         social_rate=1
                                         )

    best_individual = final_population[len(final_population) - 1]
    best_particle: Particle = best_individual.candidate
    logger.info('Fittest individual: %s', best_individual)

    # Plot the best location found
    end = Circle(best_particle.current_position, RESOURCE_RANGE, facecolor="purple", alpha=0.5)
    ax.add_patch(end)

    for individual in final_population:
        particle: Particle = individual.candidate
        # Extrapolate two arrays with x and y points with all the movements of the particle
        x, y = zip(*particle.movements)
        # Plot the list of points
        plot = ax.plot(x, y)
        logger.debug("x movements %d", len(x))

        # Plot arrows for point to point
        ax.quiver(x[:-1], y[:-1],
                  np.subtract(x[1:], x[:-1]),
                  np.subtract(y[1:], y[:-1]),
                  scale_units='xy',
                  angles='xy',
                  scale=1,
                  width=0.005,
                  color=plot[0].get_color(),
                  alpha=0.3)

    # Show the fitness value
    ax.annotate("{:.0f}".format(best_individual.fitness), best_particle.current_position)
    matplotlib.pyplot.show(block=True)
