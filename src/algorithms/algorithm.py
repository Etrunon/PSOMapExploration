from typing import Tuple

import numpy as np
import logging


from src import configuration
from src.data_structures import Map
from src.data_structures.Particle import Particle

logger = logging.getLogger(__name__)


class Algorithm:
    memo = None

    def __init__(self, map: Map) -> None:
        v_eval = np.vectorize(self.compute_score)

        self.memo = v_eval(map.resource_map)

    def evaluator(self, particle: Particle, map: Map) -> float:
        raise Exception("Don't call the base method!")

    def compute_score(self, particle: Particle, map: Map):
        raise Exception("Don't call the base method!")

    def generate(self, map: Map, starting_base: Tuple[int, int], resource_half_square, random) -> Particle:
        raise Exception("Don't call the base method!")
