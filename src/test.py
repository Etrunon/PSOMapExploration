import sys
from random import Random

from src.algo1 import generate, evaluator
from src.data_structures.Map import Map

image_name = sys.argv[1]

map = Map(image_name)

rand: Random = Random()
rand.seed(1)

x = generate(map, resource_half_square=5, random=rand, starting_base=(0, 0))  # ToDo mettere starting base casuale?
score = evaluator(x, map)
print("x: " + str(x))
print("score x: " + str(score))

print(str())
