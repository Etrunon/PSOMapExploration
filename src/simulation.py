from src import configuration
from src.main import main

parameters = {
    "resource_range": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    "cognitive_rate": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    "social_rate": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    "inertia_rate": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    "maximum_velocity": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    "population_size": {
        "lowerBound": 0,
        "upperBound": 200,
        "increment": 50
    },
    # TODO: variance threshold?
}

for a in range(parameters["resource_range"]["lowerBound"], parameters["resource_range"]["upperBound"],
               parameters["resource_range"]["increment"]):
    for b in range(parameters["cognitive_rate"]["lowerBound"], parameters["cognitive_rate"]["upperBound"],
                   parameters["cognitive_rate"]["increment"]):
        for c in range(parameters["social_rate"]["lowerBound"], parameters["social_rate"]["upperBound"],
                       parameters["social_rate"]["increment"]):
            for d in range(parameters["inertia_rate"]["lowerBound"], parameters["inertia_rate"]["upperBound"],
                           parameters["inertia_rate"]["increment"]):
                for e in range(parameters["maximum_velocity"]["lowerBound"],
                               parameters["maximum_velocity"]["upperBound"],
                               parameters["maximum_velocity"]["increment"]):
                    for f in range(parameters["population_size"]["lowerBound"],
                                   parameters["population_size"]["upperBound"],
                                   parameters["population_size"]["increment"]):
                        # Update configurations
                        configuration.RESOURCE_RANGE = a
                        configuration.COGNITIVE_RATE = b
                        configuration.SOCIAL_RATE = c
                        configuration.INERTIA_RATE = d
                        configuration.MAXIMUM_VELOCITY = e
                        configuration.POPULATION_SIZE = f

                        for i in range(0, 9):
                            best = main()

                            # write to fail
