import logging

import coloredlogs
import numpy as np
from PIL import Image

from src.configuration import IMAGE_NAME

RESOURCE_COLOURS = {
    "coal": [0, 0, 0],
    "iron": [104, 132, 146],
    "copper": [203, 97, 53],
    "stone": [174, 154, 107],
    "uranium": [0, 177, 0]
}

logger = logging.getLogger(__name__)
resources_matrix = None


def __highlight_resource(image, resource, resource_priority):
    # reference global matrix
    global resources_matrix

    resource_colour = RESOURCE_COLOURS[resource]

    # Scan the image looking for the color
    for x in range(0, image.size[0]):
        for y in range(0, image.size[1]):
            pixel = image.getpixel((x, y))
            if pixel[0] == resource_colour[0] and pixel[1] == resource_colour[1] and pixel[2] == resource_colour[2]:
                resources_matrix[x][y] = resource_priority


def detect_resource(image_name) -> np.ndarray:
    """
    Scan the input matrix looking for the different resources: coal, iron, copper and uranium.
    Foreach of them create a new image with everything black and only the relevant resource spots white (to ease the
    job of blob_detection). Then it runs blob_detection and save the result on a new matrix.
    Save the result matrix on file the first time, return it if it has already been computed.
    :param image_name: name of the image
    :return: output matrix
    """

    global resources_matrix

    # Retrieve the image and convert it to an array
    img = Image.open('data/examples/' + image_name)
    cached_matrix_path = 'data/cached_matrices/{}_resources.npy'.format(image_name.replace('.png', ''))

    try:
        resources_matrix = np.load(cached_matrix_path)
        logger.info('Loaded matrix from cached file')
    except IOError:

        logger.info('Resource matrix file does not exist or cannot be read.')
        # Init matrix to zeroes
        resources_matrix = np.zeros([img.size[0], img.size[1]], dtype=np.dtype('uint'))

        # Find all resources and load them on the resource matrix
        for resource in RESOURCE_COLOURS:
            __highlight_resource(img, resource, 1)
            logger.info('End processing of resource %s', resource)

        np.save(cached_matrix_path, resources_matrix)
        logger.info('Resource processing completed!')
    finally:
        return resources_matrix


if __name__ == "__main__":
    # Setup colored logs
    coloredlogs.install(level='INFO', style='{', fmt='{name:15s} {levelname} {message}')

    matrix = detect_resource(IMAGE_NAME)
