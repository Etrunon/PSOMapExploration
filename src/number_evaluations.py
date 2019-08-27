import csv
import os
from random import Random
from typing import List

from src import main
from src.configuration import MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY, RESOURCE_RANGE, \
    INERTIA_RATE, COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE

RUNS = int(os.environ.get("RUNS", 100))

if __name__ == '__main__':

    results: List = []

    for i in range(RUNS):
        # Initialize the random seed
        rand = Random()

        result = main.main(rand, MIN_GENERATIONS, MAX_GENERATIONS, TERMINATION_VARIANCE, MAXIMUM_VELOCITY,
                           RESOURCE_RANGE, INERTIA_RATE,
                           COGNITIVE_RATE, SOCIAL_RATE, POPULATION_SIZE, False)

        particle = result['particle']

        results.append(
            {'duration': result['duration'],
             'evaluations': result['evaluations'],
             'generations': result['generations'],
             'fitness': particle.best_fitness,
             'position_x': particle.best_position[0],
             'position_y': particle.best_position[1]}
        )

    fields = ['duration', 'evaluations', 'generations', 'fitness', 'position_x', 'position_y']
    with open('data/evaluations.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(results)
