import logging
import string
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def load_cache(path: string, fallback: Callable, *args) -> np.ndarray:
    try:
        resources = np.load(path)
        logger.info("Using cached matrix for path %s", path)
    except IOError:

        logger.info('Resource matrix file does not exist or cannot be read (path %s). Invoke fallback function.', path)
        resources = fallback(*args)

        np.save(path, resources)
        logger.info('Resource processing completed (path %s)!', path)

    return resources
