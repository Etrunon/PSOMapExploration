import logging
from _random import Random

from src.configuration import IMAGE_NAME, RESOURCE_RANGE, CITY_POSITION
from src.data_structures.Particle import Particle
from src.preprocess_utilities import load_resource_count
from src.preprocess_utilities.calculate_resources import calculate_resources

logger = logging.getLogger(__name__)


class Algorithm:
    memo = None

    resource_count_path = ''

    def __init__(self) -> None:
        self.resource_count_path = 'data/cached_matrices/{}_resource_count_{}_{}x{}.npy'. \
            format(IMAGE_NAME.replace('.png', ''), RESOURCE_RANGE, CITY_POSITION[0], CITY_POSITION[1])
        self.resource_count_matrix = load_resource_count.load_resource_count(self.resource_count_path,
                                                                             calculate_resources)

        # v_eval = np.vectorize(self.compute_score)
        # self.memo = v_eval(map.resource_map)

    def evaluator(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def compute_score(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def generate_particle(random: Random, args) -> Particle:
        raise Exception("Don't call the base method!")
