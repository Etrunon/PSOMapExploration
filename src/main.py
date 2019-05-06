import sys
from random import Random
from time import time

import inspyred
import matplotlib.pyplot
import numpy as np
from PIL import Image
from inspyred.ec import Individual
from inspyred.swarm import PSO
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src import pso
from src.Map import Map
from src.Particle import Particle


def custom_observer(population, num_generations, num_evaluations, args):
    best = max(population)
    print('Generations: {0}  Evaluations: {1}  Best: {2}'.format(num_generations, num_evaluations, str(best)))


def particle_generator(random, args):
    chromosome = []
    bounder = args["_ec"].bounder
    # The constraints are as follows:
    #             orbital   satellite   boost velocity      initial y
    #             height    mass        (x,       y)        velocity
    # for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
    #     chromosome.append(random.uniform(lo, hi))
    random_point = (random.randint(0, world_map.map_dim[0]), random.randint(0, world_map.map_dim[1]))
    return Particle(random_point, 0, resource_range=RESOURCE_RANGE, starting_base=STARTING_POSITION)


def fitness_evaluator(candidates, args):
    fitness = []
    for particle in candidates:
        # orbital_height = chromosome[0]
        # satellite_mass = chromosome[1]
        # boost_velocity = (chromosome[2], chromosome[3])
        # initial_y_velocity = chromosome[4]

        score = pso.evaluator(particle, world_map)
        fitness.append(score)

    return fitness


def variator(random, candidates, args):
    algorithm: PSO = args["_ec"]
    inertia = args.setdefault('inertia', 0.5)
    cognitive_rate = args.setdefault('cognitive_rate', 2.1)
    social_rate = args.setdefault('social_rate', 2.1)

    if len(algorithm.archive) == 0:
        algorithm.archive = algorithm.population[:]
    if len(algorithm._previous_population) == 0:
        algorithm._previous_population = algorithm.population[:]

    neighbors = algorithm.topology(algorithm._random, algorithm.archive, args)
    offspring = []

    x: Individual
    xprev: Individual
    for x, xprev, pbest, hood in zip(algorithm.population,
                                     algorithm._previous_population,
                                     algorithm.archive,
                                     neighbors):
        nbest = max(hood)

        # for xi, xpi, pbi, nbi in zip(x.candidate, xprev.candidate,
        #                              pbest.candidate, nbest.candidate):
        #     value = (xi + inertia * (xi - xpi) +
        #              cognitive_rate * random.random() * (pbi - xi) +
        #              social_rate * random.random() * (nbi - xi))
        #     particle.append(value)
        # particle = algorithm.bounder(particle, args)
        # offspring.append(particle)

        particle: Particle = x.candidate
        previous_particle: Particle = xprev.candidate
        best_particle: Particle = pbest.candidate
        best_neighbour: Particle = nbest.candidate

        new_position = (
                particle.current_position + inertia * (particle.current_position - previous_particle.current_position) +
                cognitive_rate * random.random() * (best_particle.current_position - particle.current_position) +
                social_rate * random.random() * (best_neighbour.current_position - particle.current_position)
        )

        new_position_bounded = algorithm.bounder(new_position, args)
        new_position_bounded = new_position_bounded.astype(int)  # cast to int
        particle.move_to(new_position_bounded)

        offspring.append(particle)

    return offspring


RESOURCE_RANGE = 30
STARTING_POSITION = (0, 0)

if __name__ == "__main__":
    rand = Random()
    rand.seed(int(time()))

    image_name = sys.argv[1]
    world_map = Map(image_name)

    # matplotlib.pyplot.imshow(world_map.resource_map)

    algorithm = inspyred.swarm.PSO(rand)
    algorithm.terminator = inspyred.ec.terminators.evaluation_termination
    # algorithm.observer = [inspyred.ec.observers.file_observer, inspyred.ec.observers.plot_observer, custom_observer]
    algorithm.observer = [inspyred.ec.observers.plot_observer, custom_observer]

    algorithm.variator = variator
    # algorithm.topology = inspyred.swarm.topologies.ring_topology
    algorithm.topology = inspyred.swarm.topologies.star_topology

    final_pop = algorithm.evolve(generator=particle_generator,
                                 evaluator=fitness_evaluator,
                                 pop_size=5,
                                 maximize=False,
                                 bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                 # neighborhood_size=5,
                                 max_evaluations=100,
                                 # statistics_file=stat_file,
                                 # individuals_file=ind_file)
                                 inertia=0.5
                                 )

    best = final_pop[0]
    best_particle: Particle = best.candidate
    print('\nFittest individual:')
    print(best)
    figure: Figure = matplotlib.pyplot.figure(2)
    matplotlib.pyplot.imshow(Image.open('data/examples/' + image_name))
    ax: Axes = figure.add_subplot(111)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.use_sticky_edges = False

    end = Circle(best_particle.current_position, RESOURCE_RANGE, facecolor="purple", alpha=0.5)
    ax.add_patch(end)

    start = Circle(STARTING_POSITION, 10, facecolor="red", alpha=1)
    ax.add_patch(start)

    # ax.plot(STARTING_POSITION, "ro")

    for pop in final_pop:
        particle = pop.candidate
        x, y = zip(*particle.movements)
        plot = ax.plot(x, y, ".")

        # ax.arrow(0, f(0), 0.01, f(0.01) - f(0), shape='full', lw=0, length_includes_head=True, head_width=.05)
        ax.quiver(x[:-1], y[:-1], np.subtract(x[1:], x[:-1]), np.subtract(y[1:], y[:-1]), scale_units='xy', angles='xy',
                  scale=1, width=0.005, color=plot[0].get_color(), alpha=0.3)

    pyplot.grid()
    # ax.plot(best_particle.current_position[0], best_particle.current_position[1], "or")
    matplotlib.pyplot.show(block=True)
