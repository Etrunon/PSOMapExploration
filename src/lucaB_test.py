import numpy as np


def out_of_map(position: np.ndarray) -> bool:
    """
    Subtract the resource range, so that each point having true as output, has a relevant square around it.
    If it did not subtract the range from the then it would be possible to compute the value for point 1,1 even
    though it would have only 51 x 51 points around, instead of 100x100.

    Returns:
        bool: true if the position is inside the map, false otherwise
    """
    x = position[0]
    y = position[1]

    if RESOURCE_RANGE < x < map_dim[0] - RESOURCE_RANGE:
        if RESOURCE_RANGE < y < map_dim[1] - RESOURCE_RANGE:
            return True

    return False


RESOURCE_RANGE = 5
map_dim = (20, 20)

pos = np.array((1, 16))
print("out_of_map: " + str(out_of_map(pos)))
