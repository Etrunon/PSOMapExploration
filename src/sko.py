import logging
import multiprocessing
import os
import sys
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

from src.configuration import MIN_GENERATIONS
from src.main import main

space = [
    # Integer(1, 200, name='min_generations'),
    Integer(1, 500, name='termination_variance'),
    Integer(30, 200, name='maximum_velocity'),
    Integer(10, 200, name='max_generations'),
    Integer(50, 100, name='resource_range'),
    Real(0, 1, name='cognitive_rate'),
    Real(0, 1, name='inertia_rate'),
    # Real(0, 1, name='social_rate'),
    # Integer(1, 10, name='population_size')
]

rand = Random()

RESULT_FILENAME = 'data/hyperparameters/result.skopt.gz'
CHECKPOINT_FILENAME = "data/hyperparameters/checkpoint.pkl"
MINIMIZE_CALLS = int(os.environ.get("MINIMIZE_CALLS", 10))
AGGREGATED_PARTICLES = int(os.environ.get("AGGREGATED_PARTICLES", 10))
PARALLEL_COUNT = multiprocessing.cpu_count()
SKIP_TO_RESULT = os.environ.get("SKIP_TO_RESULT", "false") == "true"
timing: List[float] = []

logger = logging.getLogger(__name__)


@use_named_args(space)
def objective(**kwargs):
    logger.info("Calling objective function with args %s", kwargs)

    cognitive_rate = kwargs['cognitive_rate']
    # social_rate = kwargs['social_rate']
    social_rate = 2 - cognitive_rate
    kwargs['social_rate'] = social_rate
    inertia = kwargs['inertia_rate']

    if (cognitive_rate + social_rate) / 2 - 1 < 0 or (cognitive_rate + social_rate) / 2 - 1 >= inertia or inertia >= 1:
        logger.warning("velocity parameters are wrong")
        return sys.maxsize

    # Save start time
    start = time.time()

    # evaluate points in parallel
    particles = Parallel(n_jobs=PARALLEL_COUNT, verbose=51)(
        delayed(main)(None, **kwargs, min_generations=MIN_GENERATIONS, show_gui=False) for i in
        range(AGGREGATED_PARTICLES)
    )

    best_fitnesses: List[float] = list(map(lambda result: result['particle'].best_fitness, particles))

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
    result = None

    try:
        result = load(RESULT_FILENAME)

    except IOError:

        # Try to load the checkpoint and if found, use it as stating point
        try:
            checkpoint = load(CHECKPOINT_FILENAME)

            if not SKIP_TO_RESULT:

                x0 = checkpoint.x_iters
                y0 = checkpoint.func_vals

                # Number of previous iterations
                previous_runs = len(y0)

                MINIMIZE_CALLS = MINIMIZE_CALLS - previous_runs
                logger.info("Found checkpoint, resuming optimization from iteration %d. Remaining %d", previous_runs,
                            MINIMIZE_CALLS)

                result = gp_minimize(objective, space,
                                     x0=x0,
                                     y0=y0,
                                     n_calls=MINIMIZE_CALLS,
                                     n_points=10,
                                     verbose=True,
                                     n_random_starts=0,
                                     callback=[checkpoint_saver]
                                     )
            else:
                result = load(CHECKPOINT_FILENAME)

        except IOError:
            # There is no checkpoint file, start from scratch
            pass

        if not result:
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
