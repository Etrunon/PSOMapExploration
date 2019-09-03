import os
import time

# Look for these parameters reading them as environment variables.
# If the variable is not found, use the given number as default value

# Name of the map image, without the folder part. For instance `map6.png`
IMAGE_NAME = os.environ["IMAGE_NAME"]

# Position of the starting base for the colony, in x and y coordinates.
# Take into account that the origin (0,0) is in the upper left corner of the image
CITY_POSITION = (int(os.environ.get("CITY_POSITION_X", 175)), int(os.environ.get("CITY_POSITION_Y", 450)))

POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", 16))
RESOURCE_RANGE = int(os.environ.get("RESOURCE_RANGE", 100))

MIN_GENERATIONS = int(os.environ.get("MIN_GENERATIONS", 1))
MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", 154))
MAX_EVALUATIONS = int(os.environ.get("MAX_EVALUATIONS", POPULATION_SIZE * MAX_GENERATIONS))
TERMINATION_VARIANCE = int(os.environ.get("TERMINATION_VARIANCE", 230))

COGNITIVE_RATE = float(os.environ.get("COGNITIVE_RATE", 0.40))
SOCIAL_RATE = float(os.environ.get("SOCIAL_RATE", 1.6))
INERTIA_RATE = float(os.environ.get("INERTIA_RATE", 0.78))

MAXIMUM_VELOCITY = int(os.environ.get("MAXIMUM_VELOCITY", 137))

# Whether to run headless or not. By default show GUI
SHOW_GUI = os.environ.get("SHOW_GUI", "true") == "true"

# Read seed from env variable, or use current time as seed
RANDOM_SEED = int(os.environ.get("SEED", time.time()))

# Use parallel computation, as provided by inspyred.
# By default it is false, since we found a lot of overhead with it.
PARALLELIZE = os.environ.get("PARALLELIZE", "false") == "true"

# CSV file where we save the benchmark results
CSV_FILE = 'data/evaluations_{}_{}_{}.csv'.format(IMAGE_NAME.replace(".png", ""), CITY_POSITION[0], CITY_POSITION[1])

# Calculate the best fitness manually and show the result on the map
CALCULATE_FITNESS_MANUALLY = False
