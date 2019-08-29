import logging
import time
from random import Random
from typing import Dict, Union

import coloredlogs
import inspyred
import matplotlib.pyplot
import numpy as np
from PIL import Image
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from src.algorithms.algo1 import Algo1
from src.configuration import CITY_POSITION, POPULATION_SIZE, COGNITIVE_RATE, INERTIA_RATE, \
    SOCIAL_RATE, IMAGE_NAME, SHOW_GUI, MIN_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, MAX_GENERATIONS, \
    RESOURCE_RANGE, MAX_EVALUATIONS, RANDOM_SEED, PARALLELIZE
from src.data_structures.Map import Map
from src.data_structures.Particle import Particle
from src.data_structures.custom_pso import log_observer, CustomPSO

logger = logging.getLogger(__name__)

# Set the theme used by matplotlib
matplotlib.pyplot.style.use("seaborn-bright")


def main(rand: Random, min_generations: int, max_generations: int, termination_variance: int, maximum_velocity: int,
         resource_range: int, inertia_rate: float, social_rate: float, cognitive_rate: float, population_size=10,
         show_gui=True, custom_observer=None) -> Dict[str, Union[float, Particle]]:
    # Save start time
    start = time.time()

    # Observers (custom logger that are notified while the algorithm runs)
    observers = [log_observer]

    if custom_observer:
        observers.append(custom_observer)

    # ######################################
    # #  Plot part   #######################
    # ######################################
    if show_gui:
        # Set the backend used by matplotlib
        matplotlib.use("Qt5Agg")

        # Create a figure, because inspyred library already creates one
        figure: Figure = matplotlib.pyplot.figure(2)

        figure.suptitle(IMAGE_NAME)

        # Plot a grid in the figure
        pyplot.grid()

        # Use the map as background
        matplotlib.pyplot.imshow(Image.open('data/examples/{}'.format(IMAGE_NAME)))

        # Save a reference to the axes object
        ax: Axes = figure.add_subplot(1, 1, 1)
        # Force an aspect for the axes
        ax.set_aspect('equal')

        # Plot the starting position as a red circle
        starting_city = Circle(CITY_POSITION, 10, facecolor="red", alpha=1)
        ax.add_patch(starting_city)

        # Use minimal padding inside the figure
        matplotlib.pyplot.tight_layout(pad=0)

        # Create a new figure, to be used by inspyred plot_observer
        figure_observer: Figure = matplotlib.pyplot.figure(1)

        # Add plot observer to draw the fitness graph
        observers.append(inspyred.ec.observers.plot_observer)

    # ######################################
    # #  Swarm part  #######################
    # ######################################
    world_map = Map(IMAGE_NAME)

    algorithm = Algo1(world_map, maximum_velocity, resource_range, save_history=show_gui)

    # initialize random if not provided
    if not rand:
        rand = Random()

    # Instantiate the custom PSO instance with the specific algorithm
    custom_pso = CustomPSO(rand, algorithm, min_generations, max_generations, termination_variance, maximum_velocity)

    # Set the observers
    custom_pso.observer = observers

    if PARALLELIZE:
        # Run the PSO algorithm using the parallel capabilities of inspyred
        final_population = custom_pso.evolve(generator=algorithm.generate_particle,
                                             evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                                             mp_evaluator=custom_pso.evaluate_particles,
                                             pop_size=population_size,
                                             maximize=False,
                                             bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                             # neighborhood_size=5,
                                             max_evaluations=MAX_EVALUATIONS,
                                             # statistics_file=stat_file,
                                             # individuals_file=ind_file)
                                             inertia=inertia_rate,
                                             cognitive_rate=cognitive_rate,
                                             social_rate=social_rate
                                             )
    else:
        # Run the PSO algorithm
        final_population = custom_pso.evolve(generator=algorithm.generate_particle,
                                             # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                                             evaluator=custom_pso.evaluate_particles,
                                             # mp_evaluator=fitness_evaluator,
                                             pop_size=population_size,
                                             maximize=False,
                                             bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                             # neighborhood_size=5,
                                             max_evaluations=MAX_EVALUATIONS,
                                             # statistics_file=stat_file,
                                             # individuals_file=ind_file)
                                             inertia=inertia_rate,
                                             cognitive_rate=cognitive_rate,
                                             social_rate=social_rate
                                             )

    for individual in final_population:
        if world_map.best_fitness >= individual.candidate.best_fitness:
            best_particle = individual.candidate

    logger.info('Fittest individual: \n%s', best_particle)

    # Save the duration of the run
    end = time.time()
    duration = end - start

    if show_gui:
        # Plot the best location found
        best_position = (best_particle.best_position[0], best_particle.best_position[1])

        rectangle = Rectangle((best_position[0] - resource_range, best_position[1] - resource_range),
                              resource_range * 2, resource_range * 2, facecolor="purple", alpha=0.4)
        # end = Circle(best_position, resource_range, facecolor="purple", alpha=0.5)
        ax.add_patch(rectangle)
        # ax.add_patch(end)

        # Show the best fitness value
        ax.annotate("{:.0f}".format(best_particle.best_fitness), best_position, color='white',
                    fontsize='x-large', fontweight='bold')

        for individual in final_population:
            particle: Particle = individual.candidate
            # Extrapolate two arrays with x and y points with all the movements of the particle
            x, y = zip(*particle.movements)

            # Plot the list of points
            plot = ax.plot(x, y, linewidth=0.2, label=particle.id)

            logger.debug("x movements %d", len(x))

            # Plot arrows for point to point
            ax.quiver(x[:-1], y[:-1],
                      np.subtract(x[1:], x[:-1]),
                      np.subtract(y[1:], y[:-1]),
                      scale_units='xy',
                      angles='xy',
                      scale=10,
                      width=0.005,
                      color=plot[0].get_color(),
                      alpha=0.3)

        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', markerscale=10)

        figManager = figure.canvas.manager.window.showMaximized()

        matplotlib.pyplot.show(block=True)

    return {'particle': best_particle, 'duration': duration, 'evaluations': custom_pso.num_evaluations,
            'generations': custom_pso.num_generations}


if __name__ == "__main__":
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    logger.info('Seed: %d', RANDOM_SEED)

    # Initialize the random seed
    rand = Random(RANDOM_SEED)

    main(rand, MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, RESOURCE_RANGE, INERTIA_RATE,
         SOCIAL_RATE, COGNITIVE_RATE, POPULATION_SIZE, SHOW_GUI)
