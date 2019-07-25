import logging
import multiprocessing
import os
import time
from random import Random
from typing import List

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_evaluations, plot_objective
from skopt.space import Integer, Real
from skopt.utils import use_named_args, dump, load

from src.main import main

space = [
    # Integer(1, 200, name='min_generations'),
    Integer(1, 200, name='termination_variance'),
    Integer(2, 200, name='maximum_velocity'),
    Integer(1, 1000, name='max_generations'),
    Integer(10, 100, name='resource_range'),
    Real(0, 1, name='cognitive_rate'),
    Real(0, 1, name='inertia_rate'),
    Real(0, 1, name='social_rate'),
    Integer(1, 10, name='population_size')
]

rand = Random()

RESULT_FILENAME = 'data/hyperparameters/result.skopt.gz'
CHECKPOINT_FILENAME = "data/hyperparameters/checkpoint.pkl"
MINIMIZE_CALLS = int(os.environ.get("MINIMIZE_CALLS", 10))
AGGREGATED_PARTICLES = int(os.environ.get("AGGREGATED_PARTICLES", 10))
PARALLEL_COUNT = multiprocessing.cpu_count()

timing: List[float] = []

logger = logging.getLogger(__name__)


@use_named_args(space)
def objective(**kwargs):
    logger.info("Calling objective function with args %s", kwargs)

    # Save start time
    start = time.time()

    # evaluate points in parallel
    particles = Parallel(n_jobs=PARALLEL_COUNT, verbose=51)(
        delayed(main)(rand, **kwargs, min_generations=200, show_gui=False) for i in
        range(AGGREGATED_PARTICLES)
    )

    best_fitnesses: List[float] = list(map(lambda particle: particle.best_fitness, particles))

    # Save the duration of the run
    end = time.time()
    objective_duration = end - start
    timing.append(objective_duration)

    logger.info(
        'Objective function aggregated %d particles and returns the mean %d',
        AGGREGATED_PARTICLES,
        np.mean(best_fitnesses)
    )

    return np.mean(best_fitnesses)


if __name__ == '__main__':
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    checkpoint_saver = CheckpointSaver(CHECKPOINT_FILENAME,
                                       compress=9)  # keyword arguments will be passed to `skopt.dump`

    try:
        result = load(RESULT_FILENAME)

    except IOError:

        # Try to load the checkpoint and if found, use it as stating point
        try:
            checkpoint = load(CHECKPOINT_FILENAME)

            logger.info("Found checkpoint, resuming optimization")

            x0 = checkpoint.x_iters
            y0 = checkpoint.func_vals

            result = gp_minimize(objective, space,
                                 x0=x0,
                                 y0=y0,
                                 n_calls=MINIMIZE_CALLS,
                                 n_points=10,
                                 verbose=True,
                                 callback=[checkpoint_saver]
                                 )

        except IOError:
            # There is no checkpoint file, start from scratch
            pass

        logger.info("No checkpoint file available")
        result = gp_minimize(objective, space,
                             n_calls=MINIMIZE_CALLS,
                             n_points=10,
                             verbose=True,
                             callback=[checkpoint_saver]
                             )

        dump(result, filename=RESULT_FILENAME)

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
