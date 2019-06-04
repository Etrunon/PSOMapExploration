import logging
from typing import Tuple

from src.configuration import IMAGE_NAME
from src.data_structures import Map
from src.data_structures.Particle import Particle
from src.preprocess_utilities import utilities
from src.preprocess_utilities.calculate_resources import calculate_resources

logger = logging.getLogger(__name__)


class Algorithm:
    memo = None

    cached_matrix_path = ''

    def __init__(self, map: Map) -> None:
        self.cached_matrix_path = 'data/cached_matrices/' + IMAGE_NAME.replace('.png', '') + '_resource_count.npy'
        self.cached_resource_count = utilities.load_cache(self.cached_matrix_path, calculate_resources, map)

        # v_eval = np.vectorize(self.compute_score)
        # self.memo = v_eval(map.resource_map)

    def evaluator(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def compute_score(self, particle: Particle) -> float:
        raise Exception("Don't call the base method!")

    def generate(self, map: Map, starting_base: Tuple[int, int], resource_half_square, random) -> Particle:
        raise Exception("Don't call the base method!")
