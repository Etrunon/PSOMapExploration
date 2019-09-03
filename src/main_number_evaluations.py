import csv
import multiprocessing
import os
from typing import List

from joblib import Parallel, delayed

from src import main
from src.configuration import MIN_GENERATIONS, CITY_POSITION, CSV_FILE

############################
# Create a CSV files with performance results of a lot of runs fo the algorithm on the same map
############################

RUNS = int(os.environ.get("RUNS", 500))
PARALLEL_COUNT = multiprocessing.cpu_count()

if __name__ == '__main__':

    results: List = []

    parallel_results = Parallel(n_jobs=PARALLEL_COUNT, verbose=51)(
        delayed(main.main)(None,
                           MIN_GENERATIONS, 154,
                           230,
                           137,
                           100,
                           0.78, 1.6, 0.40,
                           16,
                           False)
        for i in range(RUNS)
    )

    for result in parallel_results:
        particle = result['particle']

        results.append(
            {
                'duration': result['duration'],
                'evaluations': result['evaluations'],
                'generations': result['generations'],
                'fitness': particle.best_fitness,
                'position_x': particle.best_position[0],
                'position_y': particle.best_position[1],
                'city_position_x': CITY_POSITION[0],
                'city_position_y': CITY_POSITION[1]
            }
        )

    fields = ['duration',
              'evaluations',
              'generations',
              'fitness',
              'position_x',
              'position_y',
              'city_position_x',
              'city_position_y']

    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(results)
