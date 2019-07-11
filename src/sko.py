from random import Random

import coloredlogs
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from src.data_structures.Map import world_map
from src.main import main

space = [
    Integer(1, 200, name='min_generations'),
    Integer(1, 200, name='min_generations')

]
rand = Random()


@use_named_args(space)
def objective(**kwargs):

    world_map.best_fitness = 10000

    best_particle = main(rand, show_gui=False, **kwargs)

    return best_particle.best_fitness


if __name__ == '__main__':
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    res = gp_minimize(objective, space)

    print("Best score=%.4f" % res.fun)
    print("min_generations: %s" % res.x[0])
