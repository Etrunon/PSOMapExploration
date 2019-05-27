import logging
import sys
from random import Random
from typing import List, Dict

import coloredlogs
import inspyred
import matplotlib.pyplot
import numpy as np
from PIL import Image
from inspyred.ec import Individual
from inspyred.swarm import PSO
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src import algo1
from src.data_structures.Map import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)

# Set the theme used by matplotlib
matplotlib.pyplot.style.use("seaborn-bright")
# Set the backend used by matplotlib
matplotlib.use("Qt5Agg")


def custom_observer(population, num_generations, num_evaluations, args) -> None:
    """
    Log the best individual for each generation
    """

    best = max(population)
    logger.debug('Generations: %d  Evaluations: %d  Best: %s', num_generations, num_evaluations, best)


def particle_generator(random: Random, args):
    """
    Position the particles randomly on the map
    """
    # bounder: Bounder = args["_ec"].bounder
    random_point = (random.randint(0, world_map.map_dim[0]), random.randint(0, world_map.map_dim[1]))
    return Particle(random_point, 0, resource_range=RESOURCE_RANGE, starting_base=STARTING_POSITION)


def fitness_evaluator(candidates: List[Particle], args):
    fitness = []
    for particle in candidates:
        score = algo1.evaluator(particle, world_map)
        fitness.append(score)

    return fitness


def variator(random: Random, candidates: List[Particle], args):
    algorithm: PSO = args["_ec"]
    inertia = args.setdefault('inertia', 0.5)
    cognitive_rate = args.setdefault('cognitive_rate', 2.1)
    social_rate = args.setdefault('social_rate', 2.1)

    if len(algorithm.archive) == 0:
        algorithm.archive = algorithm.population[:]
    if len(algorithm._previous_population) == 0:
        algorithm._previous_population = algorithm.population[:]

    neighbors = algorithm.topology(algorithm._random, algorithm.archive, args)
    offspring: List[Particle] = []

    x: Individual
    xprev: Individual  # represents the particle at the previous iteration
    for x, xprev, pbest, neighbor in zip(algorithm.population,
                                         algorithm._previous_population,
                                         algorithm.archive,
                                         neighbors):
        nbest = max(neighbor)

        particle: Particle = x.candidate
        previous_particle: Particle = xprev.candidate
        best_particle: Particle = pbest.candidate
        best_neighbour: Particle = nbest.candidate

        # new position is obtained considering the inertia of the particle, its position wrt the best individual in the
        # swarm and the best individual among its neighbors
        new_position = (
                particle.current_position + inertia * (particle.current_position - previous_particle.current_position) +
                cognitive_rate * random.random() * (best_particle.current_position - particle.current_position) +
                social_rate * random.random() * (best_neighbour.current_position - particle.current_position)
        )

        # the bounder filters out unwanted position values
        new_position_bounded = algorithm.bounder(new_position, args)
        new_position_bounded = new_position_bounded.astype(int)  # cast to int
        particle.move_to(new_position_bounded)

        offspring.append(particle)

    return offspring


def evaluation_termination(population: List[Individual], num_generations: int, num_evaluations: int,
                           args: Dict) -> bool:
    fitnesses = []

    for index, p in enumerate(population):
        fitnesses = np.insert(fitnesses, index, p.fitness)
    variance = np.var(fitnesses)

    if variance < TERMINATION_VARIANCE:
        return True
    else:
        return False


RESOURCE_RANGE = 100
STARTING_POSITION = (0, 0)
POPULATION_SIZE = 2
TERMINATION_VARIANCE = 5000  # TODO: find optimal value


class CustomPSO(PSO):
    """Custom implementation of the PSO object, to allow for serialization and de-serialization for multiprocess support
    """

    def __init__(self, random):
        super().__init__(random)

    def __getstate__(self):
        """ Invoked by Python to save the object for serialization
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        """ Invoked by Python to restore the object from deserialization
        """
        self.__dict__.update(d)


if __name__ == "__main__":
    # Setup colored logs
    coloredlogs.install(level='DEBUG', style='{', fmt='{name:15s} {levelname} {message}')

    # Initialize the random seed
    rand = Random()
    rand.seed(1)

    image_name = sys.argv[1]
    world_map = Map(image_name)

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
    algorithm = CustomPSO(rand)

    # Set custom properties for the PSO instance
    algorithm.terminator = evaluation_termination

    # Set the custom variator to move each particle in the swarm
    algorithm.variator = variator

    # Set the topology to specify how neighbours are found
    algorithm.topology = inspyred.swarm.topologies.star_topology

    # Observers (custom logger that are notified while the algorithm runs)
    algorithm.observer = [inspyred.ec.observers.plot_observer, custom_observer]

    # Create a new figure, to be used by inspyred plot_observer
    figure: Figure = matplotlib.pyplot.figure(1)

    # Run the PSO algorithm
    final_population = algorithm.evolve(generator=particle_generator,
                                        # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                                        evaluator=fitness_evaluator,
                                        # mp_evaluator=fitness_evaluator,
                                        pop_size=POPULATION_SIZE,
                                        maximize=False,
                                        bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                        # neighborhood_size=5,
                                        max_evaluations=10,
                                        # statistics_file=stat_file,
                                        # individuals_file=ind_file)
                                        inertia=0.7,
                                        cognitive_rate=0.3,
                                        social_rate=0.01
                                        )

    best_individual = final_population[0]
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
        plot = ax.plot(x, y, ".")

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
