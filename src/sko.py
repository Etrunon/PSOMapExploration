from random import Random

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
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

FILENAME = 'result.skopt.gz'


@use_named_args(space)
def objective(**kwargs):

    best_particle = main(rand, **kwargs, min_generations=200, show_gui=False)

    return best_particle.best_fitness


if __name__ == '__main__':
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    try:
        result = load(FILENAME)

    except IOError:
        result = gp_minimize(objective, space, n_calls=100, n_points=10)

        dump(result, filename=FILENAME)

    print("Best score=%.4f" % result.fun)

    [print(dimension.name) for dimension in space]

    print("%s" % result.x)

    # Set the backend used by matplotlib
    matplotlib.use("Qt5Agg")
    plot_evaluations(result, bins=10)
    plot_objective(result)
    plt.show(block=True)
