import os

RESOURCE_RANGE = 100
CITY_POSITION = (500, 600)
TERMINATION_VARIANCE = 200  # TODO: find optimal value
MIN_GENERATIONS = int(os.environ.get("MIN_GENERATIONS", 100))  # T0ODO: find optimal value
MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", 1000))  # TODO: find optimal value
IMAGE_NAME = os.environ["IMAGE_NAME"]

COGNITIVE_RATE = 0.50
SOCIAL_RATE = 0.50
INERTIA_RATE = 0.95
MAXIMUM_VELOCITY = 20

POPULATION_SIZE = 5

SHOW_GUI = os.environ.get("SHOW_GUI", "true") == "true"
