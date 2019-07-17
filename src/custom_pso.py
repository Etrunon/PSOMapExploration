import logging
import math
from random import Random
from typing import List, Dict

import numpy as np
from inspyred.ec import Individual
from inspyred.swarm import PSO

from src.algorithms.algorithm import Algorithm
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


def custom_observer(population, num_generations, num_evaluations, args) -> None:
    """
    Log the best individual for each generation
    """

    best = min(population)
    logger.debug('Generations: %d  Evaluations: %d  Best: %s', num_generations, num_evaluations, best)


class CustomPSO(PSO):
    """
    Custom implementation of the PSO object, to allow for serialization and de-serialization for multiprocess support
    """
    _algorithm: Algorithm = None

    def get_algorithm(self):
        assert self._algorithm is not None
        return self._algorithm

    def __init__(self, random, algorithm: Algorithm, min_generations: int, maximum_generations: int,
                 termination_variance: int, maximum_velocity: int):
        super().__init__(random)
        self.maximum_generations = maximum_generations
        self.maximum_velocity = maximum_velocity
        self.termination_variance = termination_variance
        self.min_generations = min_generations
        self._algorithm = algorithm

        # Set the custom variator to move each particle in the swarm
        self.variator = self.custom_variator

    def custom_terminator(self, population: List[Individual], num_generations: int, num_evaluations: int,
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

        if 0 < variance < self.termination_variance and num_generations > self.min_generations:
            logger.warning('>>>>>>>>> End for variance condition. Total Evaluation: ' + str(num_evaluations))
            return True
        elif num_generations > self.maximum_generations:
            logger.warning('>>>>>>>>> End for max generations reached. Total Evaluation: ' + str(num_evaluations))
            return True
        else:
            return False

    def custom_variator(self, random: Random, candidates: List[Particle], args: Dict) -> List[Particle]:
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

        neighbors_generator = algorithm.topology(algorithm._random, algorithm.archive, args)
        offspring: List[Particle] = []

        # noinspection PyUnusedLocal
        x: Individual
        # noinspection PyUnusedLocal
        neighbors: List[Individual]

        for x, neighbors in zip(algorithm.population, neighbors_generator):

            best_neighbour = min(neighbors, key=lambda x: x.candidate.best_fitness).candidate
            particle: Particle = x.candidate

            new_velocity = (
                    particle.velocity * inertia +
                    cognitive_rate * random.random() * (particle.best_position - particle.current_position) +
                    social_rate * random.random() * (best_neighbour.best_position - particle.current_position)
            )

            # Limit the velocity up to a maximum
            norm = np.linalg.norm(new_velocity)
            if norm > self.maximum_velocity:
                new_velocity = (new_velocity / norm) * self.maximum_velocity

            new_position = particle.current_position + new_velocity

            if not self._algorithm.world_map.is_inside_map(new_position, self._algorithm.resource_range):
                # Ricalcola un nuovo vettore velocità a caso e riprova
                inside = False
                while not inside:
                    angle = random.randint(0, 360)

                    # Rotate the vector
                    tmp_velocity_x = new_velocity[0] * math.cos(angle) - new_velocity[1] * math.sin(angle)
                    tmp_velocity_y = new_velocity[0] * math.sin(angle) + new_velocity[1] * math.cos(angle)

                    # Assign them
                    new_velocity[0] = tmp_velocity_x
                    new_velocity[1] = tmp_velocity_y

                    new_position = particle.current_position + new_velocity
                    inside = self._algorithm.world_map.is_inside_map(new_position, self._algorithm.resource_range)

            particle.move_to(new_position.astype(int))
            particle.set_velocity(new_velocity)
            offspring.append(particle)

        return offspring

    def evaluate_particles(self, candidates: List[Particle], args) -> List[float]:
        """
        Evaluate the particle considering the fitness of all the swarm particles.
        - update the best position for each particle,  if applicable
        - update the global best position, if applicable

        Returns: The list of fitness values, one for each particle
        """
        fitness: List[float] = []

        for particle in candidates:
            score = args["_ec"].get_algorithm().evaluator(particle)

            # Update the particle best fitness, if current one is better
            if score < particle.best_fitness:
                particle.best_fitness = score
                particle.best_position = particle.current_position

            # Update the global position, if the current one is better
            if score < self._algorithm.world_map.best_fitness:
                self._algorithm.world_map.best_fitness = score
                self._algorithm.world_map.best_position = particle.current_position

            fitness.append(score)

        return fitness
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
