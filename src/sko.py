import logging
import os
import time
from random import Random
from typing import List

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective
from skopt.space import Integer
from skopt.utils import use_named_args, dump, load

from src.main import main

space = [
    # Integer(1, 200, name='min_generations'),
    Integer(1, 200, name='termination_variance'),
    Integer(2, 200, name='maximum_velocity'),
    Integer(1, 1000, name='max_generations'),
    Integer(10, 100, name='resource_range')
]

rand = Random()

FILENAME = 'data/hyperparameters/result.skopt.gz'
MINIMIZE_CALLS = int(os.environ.get("MINIMIZE_CALLS", 10))
AGGREGATED_PARTICLES = int(os.environ.get("AGGREGATED_PARTICLES", 10))

timing: List[float] = []

logger = logging.getLogger(__name__)


@use_named_args(space)
def objective(**kwargs):
    aggregated_best_individuals: List[float] = []
    start = time.time()

    for i in range(0, AGGREGATED_PARTICLES):
        best_particle = main(rand, **kwargs, min_generations=200, show_gui=False)
        aggregated_best_individuals.append(best_particle.best_fitness)

    end = time.time()
    objective_duration = end - start
    timing.append(objective_duration)
    # print("skopt took %.4f seconds to execute objective" % objective_duration)
    logger.info('Objective aggregated %d particles and returns the mean %d', AGGREGATED_PARTICLES,
                np.mean(aggregated_best_individuals))

    return np.mean(aggregated_best_individuals)


if __name__ == '__main__':
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    try:
        result = load(FILENAME)

    except IOError:
        result = gp_minimize(objective, space, n_calls=MINIMIZE_CALLS, n_points=10)

        dump(result, filename=FILENAME)

    print("Best score=%.4f" % result.fun)

    [print(dimension.name) for dimension in space]

    print("%s" % result.x)

    # Set the backend used by matplotlib
    matplotlib.use("Qt5Agg")
    plot_evaluations(result, bins=10)
    plot_objective(result)

    fig, ax_lst = plt.subplots(1, 1)
    plt.plot(timing)
    plt.xlabel('average timing %.4f seconds' % np.mean(timing))
    plt.show(block=True)
