import csv
import time
from random import Random
from typing import List

from src import main
from src.configuration import MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, RESOURCE_RANGE, \
    INERTIA_RATE, COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE
from src.data_structures.Particle import Particle

timing: List[float] = []

RUNS = 100

# global variable to save evaluations
EVALUATIONS = 0


def evaluations_observer(population, num_generations, num_evaluations, args) -> None:
    global EVALUATIONS

    EVALUATIONS = num_evaluations


if __name__ == '__main__':

    results: List[Particle] = []

    for i in range(RUNS):
        # Save start time
        start = time.time()

        # Initialize the random seed
        rand = Random()

        result = main.main(rand, MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY,
                           RESOURCE_RANGE, INERTIA_RATE,
                           COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE, False, evaluations_observer)

        results.append(EVALUATIONS)

    # fig, ax_lst = plt.subplots(1, 1)
    #
    # plt.scatter(timing, fitnesses, c='r', label="Random parameters")
    # plt.scatter(optimized_timings, optimized_fitnesses, c='b', label="Optimized parameters")

    # plt.xlabel('Time (s)')
    # plt.ylabel('Fitness')
    #
    # plt.legend(loc='lower right')
    #
    # plt.show(block=True)

    with open('data/evaluations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(results)

    print(results)
