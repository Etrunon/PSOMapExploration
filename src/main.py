import sys
from random import Random
from time import time

import inspyred
import matplotlib.pyplot
from PIL import Image
from inspyred.ec import Individual
from inspyred.swarm import PSO
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
    return Particle(random_point, 0, resource_range=RESOURCE_RANGE, starting_base=(0, 0))


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

        particle.current_position = (
                particle.current_position + inertia * (particle.current_position - previous_particle.current_position) +
                cognitive_rate * random.random() * (best_particle.current_position - particle.current_position) +
                social_rate * random.random() * (best_neighbour.current_position - particle.current_position)
        )

        particle.current_position = algorithm.bounder(particle.current_position, args)
        print("New position:" + str(particle.current_position))
        particle.current_position = particle.current_position.astype(int)
        offspring.append(particle)

    return offspring


RESOURCE_RANGE = 30

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
                                 pop_size=100,
                                 maximize=False,
                                 bounder=inspyred.ec.Bounder(0, max(world_map.map_dim)),
                                 # neighborhood_size=5,
                                 max_evaluations=1000,
                                 # statistics_file=stat_file,
                                 # individuals_file=ind_file)
                                 inertia=0.8
                                 )

    best = final_pop[0]
    best_particle: Particle = best.candidate
    print('\nFittest individual:')
    print(best)
    figure: Figure = matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(Image.open('data/examples/' + image_name))
    ax: Axes = figure.add_subplot(111)
    circle = Circle(best_particle.current_position, RESOURCE_RANGE, facecolor="purple", alpha=0.5)
    ax.add_patch(circle)
    # ax.plot(best_particle.current_position[0], best_particle.current_position[1], "or")
    matplotlib.pyplot.show(block=True)
