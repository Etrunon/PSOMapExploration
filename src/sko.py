import logging
import os
import pickle
import time
from random import Random
from typing import List

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from skopt import Optimizer
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
OPTIMIZER_FILENAME = "data/hyperparameters/optimizer.pkl"

MINIMIZE_CALLS = int(os.environ.get("MINIMIZE_CALLS", 10))
AGGREGATED_PARTICLES = int(os.environ.get("AGGREGATED_PARTICLES", 10))

timing: List[float] = []

logger = logging.getLogger(__name__)


@use_named_args(space)
def objective(**kwargs):
    logger.info(kwargs)

    start = time.time()

    particles = Parallel(n_jobs=4, verbose=51)(
        delayed(main)(rand, **kwargs, min_generations=200, show_gui=False) for i in
        range(AGGREGATED_PARTICLES))  # evaluate points in parallel

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

    result = None

    try:
        result = load(RESULT_FILENAME)
        logger.info("Result file loaded")

    except IOError:
        # There is no result file
        pass

    if not result:
        # Try to load the optimizer and if found, use it instead of starting from scratch

        optimizer = None
        try:

            with open(OPTIMIZER_FILENAME, 'rb') as f:
                optimizer = pickle.load(f)
                logger.info("Cached optimizer loaded")

        except IOError:
            # There is no saved optimizer file, start from scratch
            logger.info("No cached optimizer available")
            pass

        if not optimizer:
            # There is no optimizer available, create one from scratch
            optimizer = Optimizer(
                dimensions=space,
                base_estimator='gp'
            )

        POINTS = 1
        for i in range(MINIMIZE_CALLS):
            x = optimizer.ask(n_points=POINTS)  # x is a list of n_points points

            # y = Parallel(n_jobs=4, verbose=1)(delayed(objective)(v) for v in x)  # evaluate points in parallel
            y = objective(x[0])

            # logger.info("%s %s", x, y)

            result = optimizer.tell(x, y)

            # Cache the optimizer to resume work later
            with open(OPTIMIZER_FILENAME, 'wb') as f:
                pickle.dump(optimizer, f)

        # Cache the result file
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
