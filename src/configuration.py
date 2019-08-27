import os
import time

IMAGE_NAME = os.environ["IMAGE_NAME"]
CITY_POSITION = (471, 94)

POPULATION_SIZE = 5
RESOURCE_RANGE = 40

MIN_GENERATIONS = int(os.environ.get("MIN_GENERATIONS", 100))  # T0ODO: find optimal value
MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", 1000))  # TODO: find optimal value
MAX_EVALUATIONS = int(os.environ.get("MAX_EVALUATIONS", 2000))
TERMINATION_VARIANCE = 200  # TODO: find optimal value

COGNITIVE_RATE = float(os.environ.get("COGNITIVE_RATE", 0.10))
SOCIAL_RATE = float(os.environ.get("SOCIAL_RATE", 0.001))
INERTIA_RATE = float(os.environ.get("INERTIA_RATE", 0.99))

MAXIMUM_VELOCITY = int(os.environ.get("MAXIMUM_VELOCITY", 8))

SHOW_GUI = os.environ.get("SHOW_GUI", "true") == "true"

# Read seed from env variable, or use current time as seed
RANDOM_SEED = int(os.environ.get("SEED", time.time()))
