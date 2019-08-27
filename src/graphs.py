import multiprocessing
import time
from typing import List

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src import main
from src.configuration import MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, RESOURCE_RANGE, \
    INERTIA_RATE, COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE
from src.data_structures.Particle import Particle

timing: List[float] = []
optimized_timings = []

RUNS = 100
PARALLEL_COUNT = multiprocessing.cpu_count()

if __name__ == '__main__':

    results: List[Particle] = []
    optimized_results = []

    # Save start time
    start = time.time()

    parallel_results = Parallel(n_jobs=PARALLEL_COUNT, verbose=51)(
        delayed(main.main)(None, MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY,
                           RESOURCE_RANGE,
                           INERTIA_RATE, SOCIAL_RATE, COGNITIVE_RATE,
                           POPULATION_SIZE, False)
        for i in range(RUNS)
    )

    for result in parallel_results:
        results.append(result['particle'])
        timing.append(result['duration'])


    parallel_results = Parallel(n_jobs=PARALLEL_COUNT, verbose=51)(
        delayed(main.main)(None, MIN_GENERATIONS, 168, 32, 137,
                           100,
                           0.94, 1.55, 0.45,
                           POPULATION_SIZE, False)
        for i in range(RUNS)
    )

    for result in parallel_results:
        optimized_results.append(result['particle'])
        optimized_timings.append(result['duration'])

    fitnesses = list(map(lambda particle: particle.best_fitness, results))
    optimized_fitnesses = list(map(lambda particle: particle.best_fitness, optimized_results))

    fig, ax_lst = plt.subplots(1, 1)

    plt.scatter(timing, fitnesses, c='r', label="Random parameters")
    plt.scatter(optimized_timings, optimized_fitnesses, c='b', label="Optimized parameters")

    plt.xlabel('Time (s)')
    plt.ylabel('Fitness')

    plt.legend(loc='lower right')

    plt.show(block=True)
