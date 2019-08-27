import os
import time

IMAGE_NAME = os.environ["IMAGE_NAME"]
CITY_POSITION = (175, 450)

POPULATION_SIZE = 5
RESOURCE_RANGE = 50

MIN_GENERATIONS = int(os.environ.get("MIN_GENERATIONS", 100))
MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", 1000))
MAX_EVALUATIONS = int(os.environ.get("MAX_EVALUATIONS", POPULATION_SIZE * 2000))
TERMINATION_VARIANCE = 200

COGNITIVE_RATE = float(os.environ.get("COGNITIVE_RATE", 0.9))
SOCIAL_RATE = float(os.environ.get("SOCIAL_RATE", 1.7))
INERTIA_RATE = float(os.environ.get("INERTIA_RATE", 0.8))

MAXIMUM_VELOCITY = int(os.environ.get("MAXIMUM_VELOCITY", 30))

SHOW_GUI = os.environ.get("SHOW_GUI", "true") == "true"

# Read seed from env variable, or use current time as seed
RANDOM_SEED = int(os.environ.get("SEED", time.time()))
