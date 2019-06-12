import logging
import math
from random import Random
from typing import List, Dict

import numpy as np
from inspyred.ec import Individual
from inspyred.swarm import PSO

from src.algorithms.algorithm import Algorithm
from src.configuration import RESOURCE_RANGE, STARTING_POSITION, TERMINATION_VARIANCE, MIN_GENERATION, MAX_GENERATION, \
    MAXIMUM_VELOCITY
from src.data_structures import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


def custom_terminator(population: List[Individual], num_generations: int, num_evaluations: int,
                      args: Dict) -> bool:
    """

    Returns:
        bool: True if the variance of the fitnesses of all the particles is greater than 0 and below a given threshold
        after a certain number of generations or once a certain number of generations is reached.
    """
    fitnesses = []

    for index, p in enumerate(population):
        fitnesses = np.insert(fitnesses, index, p.fitness)
    variance = np.var(fitnesses)

    if 0 < variance < TERMINATION_VARIANCE and num_generations > MIN_GENERATION:
        logger.warning('>>>>>>>>> End for variance condition.')
        return True
    elif num_generations > MAX_GENERATION:
        logger.warning('>>>>>>>>> End for max generations reached.')
        return True
    else:
        return False


def custom_observer(population, num_generations, num_evaluations, args) -> None:
    """
    Log the best individual for each generation
    """

    best = max(population)
    logger.debug('Generations: %d  Evaluations: %d  Best: %s', num_generations, num_evaluations, best)


def custom_variator(random: Random, candidates: List[Particle], args: Dict) -> List[Particle]:
    """
    Update the position of each particle in the swarm.
    This function is called by inspyred once for each generation

    Returns: A list of particle, with their positions modified

    """

    algorithm: CustomPSO = args["_ec"]
    inertia = args.setdefault('inertia', 0.5)
    cognitive_rate = args.setdefault('cognitive_rate', 2.1)
    social_rate = args.setdefault('social_rate', 2.1)

    if len(algorithm.archive) == 0:
        algorithm.archive = algorithm.population[:]
    # if len(algorithm._previous_population) == 0:
    #     algorithm._previous_population = algorithm.population[:]

    neighbors_generator = algorithm.topology(algorithm._random, algorithm.archive, args)
    offspring: List[Particle] = []

    x: Individual
    neighbors: List[Individual]
    for x, neighbors in zip(algorithm.population, neighbors_generator):
        best_neighbour = max(neighbors)

        particle: Particle = x.candidate
        best_neighbour_particle: Particle = best_neighbour.candidate

        velocity = (
                particle.velocity * inertia +
                cognitive_rate * random.random() * (particle.best_position - particle.current_position) +
                social_rate * random.random() * (best_neighbour_particle.best_position - particle.current_position)
        )

        new_position = particle.current_position + velocity
        norm = np.linalg.norm(velocity)

        # TODO: bound velocity
        if norm > MAXIMUM_VELOCITY:
            # normalization_factor = norm / MAXIMUM_VELOCITY
            velocity = (velocity / norm) * MAXIMUM_VELOCITY

        if algorithm.world_map.is_inside_map(new_position):

            random_coordinate_x = random.random() * norm
            if random.random() > 0.5:
                random_coordinate_x = - random_coordinate_x

            random_coordinate_y = math.sqrt(norm ** 2 - random_coordinate_x ** 2)

            if random.random() > 0.5:
                random_coordinate_y = - random_coordinate_y

            new_velocity = np.array([random_coordinate_x, random_coordinate_y])

            new_position = particle.current_position
            velocity = new_velocity

        particle.move_to(new_position.astype(int))
        particle.set_velocity(velocity)
        offspring.append(particle)

    return offspring


def generate_particle(random: Random, args):
    """
    Position the particle randomly on the map
    """
    # bounder: Bounder = args["_ec"].bounder
    algorithm: CustomPSO = args["_ec"]
    world_map = algorithm.get_world_map()
    random_point = (random.randint(0, world_map.map_dim[0]), random.randint(0, world_map.map_dim[1]))
    velocity = (random.randint(1, 100), random.randint(1, 100))

    return Particle(starting_position=random_point,
                    velocity=np.array(velocity, np.uintc),
                    resource_range=RESOURCE_RANGE,
                    starting_base=STARTING_POSITION)


def evaluate_particle(candidates: List[Particle], args) -> List[float]:
    """
    Evaluate the particle considering the fitness of all the swarm particles.
    - update the best position for each particle,  if applicable
    - update the global best position, if applicable

    Returns: The list of fitness values, one for each particle
    """
    fitness: List[float] = []
    world_map = args["_ec"].world_map
    for particle in candidates:
        score = args["_ec"].get_algorithm().evaluator(particle)

        # Update the particle best fitness, if current one is better
        if score < particle.best_fitness:
            particle.best_fitness = score
            particle.best_position = particle.current_position

        # Update the global position, if the current one is better
        if score < world_map.best_fitness:
            world_map.best_fitness = score
            world_map.best_position = particle.current_position

        fitness.append(score)

    return fitness


class CustomPSO(PSO):
    """
    Custom implementation of the PSO object, to allow for serialization and de-serialization for multiprocess support
    """

    world_map: Map = None
    _algorithm: Algorithm = None

    def set_world_map(self, world_map: Map):
        self.world_map = world_map

    def get_world_map(self):
        assert self.world_map is not None
        return self.world_map

    def set_algorithm(self, algorithm: Algorithm):
        self._algorithm = algorithm

    def get_algorithm(self):
        assert self._algorithm is not None
        return self._algorithm

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
