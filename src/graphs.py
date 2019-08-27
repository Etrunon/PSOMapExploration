import time
from random import Random
from typing import List

import matplotlib.pyplot as plt

from src import main
from src.configuration import MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, RESOURCE_RANGE, \
    INERTIA_RATE, COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE, SHOW_GUI
from src.data_structures.Particle import Particle

timing: List[float] = []
optimized_timings = []

RUNS = 100

if __name__ == '__main__':

    results: List[Particle] = []
    optimized_results = []

    for i in range(RUNS):
        # Save start time
        start = time.time()

        # Initialize the random seed
        rand = Random()

        result = main.main(rand, MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY,
                           RESOURCE_RANGE,
                           INERTIA_RATE, SOCIAL_RATE, COGNITIVE_RATE,
                           POPULATION_SIZE, SHOW_GUI)

        results.append(result)

        # Save the duration of the run
        end = time.time()
        duration = end - start
        timing.append(duration)

    for i in range(RUNS):
        # Save start time
        start = time.time()

        # Initialize the random seed
        rand = Random()

        result = main.main(rand,
                           MIN_GENERATIONS,
                           276,
                           133,
                           173,
                           100,
                           0.8750,
                           0.9362,
                           0.8040,
                           POPULATION_SIZE,
                           SHOW_GUI)

        optimized_results.append(result)

        # Save the duration of the run
        end = time.time()
        objective_duration = end - start
        optimized_timings.append(objective_duration)

    fitnesses = list(map(lambda particle: particle.best_fitness, results))
    optimized_fitnesses = list(map(lambda particle: particle.best_fitness, optimized_results))

    fig, ax_lst = plt.subplots(1, 1)

    plt.scatter(timing, fitnesses, c='r', label="Random parameters")
    plt.scatter(optimized_timings, optimized_fitnesses, c='b', label="Optimized parameters")

    plt.xlabel('Time (s)')
    plt.ylabel('Fitness')

    plt.legend(loc='lower right')

    plt.show(block=True)
