import logging
from _random import Random

from src.configuration import IMAGE_NAME, CITY_POSITION
from src.data_structures import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


class Algorithm:
    resource_range = 0
    resource_count_path = ''

    def __init__(self, world_map: Map, resource_range: int) -> None:
        self.world_map = world_map
        self.resource_range = resource_range

        self.resource_count_path = 'data/cached_matrices/{}_resource_count_{}_{}x{}.npy'. \
            format(IMAGE_NAME.replace('.png', ''), self.resource_range, CITY_POSITION[0], CITY_POSITION[1])
        # self.resource_count_matrix = load_resource_count.load_resource_count(self.resource_count_path,
        #                                                                      calculate_resources)

    def evaluator(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def compute_score(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def generate_particle(self, random: Random, args) -> Particle:
        raise Exception("Don't call the base method!")
