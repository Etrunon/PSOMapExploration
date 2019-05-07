import numpy as np

from src.Map import Map


class Particle:
    current_position = (None, None)
    velocity = None
    local_best = (None, None, None)
    starting_base = (None, None)
    resource_half_square = None

    def __init__(self, current_position, velocity, starting_base, resource_range):
        self.current_position = np.array([current_position[0], current_position[1]])
        self.velocity = velocity
        self.local_best = (0, 0, 0)
        self.starting_base = np.array([starting_base[0], starting_base[1]])
        self.resource_half_square = resource_range

    def count_resources(self, map: Map):
        """
        Compute the value of all resources around the square.
        The computation finds the actual bounding box around the chosen point, making sure not to get outside the
        limits of the matrix (in the case the chosen point is too close to the border).
        The idea is to check if each vertex of the bounding box is in a legal position and if not replace it with the border.
        As a note, v3 is actually not needed.
        :return: the number of resources
        """
        # v1                   v2
        #      true_v1------true_v2--------
        #      |
        #      |      cp
        #      |
        # v4   true_v4         v3

        v1 = (
        self.current_position[0] - self.resource_half_square, self.current_position[1] - self.resource_half_square)
        v2 = (
        self.current_position[0] + self.resource_half_square, self.current_position[1] - self.resource_half_square)
        v3 = (
        self.current_position[0] + self.resource_half_square, self.current_position[1] + self.resource_half_square)
        v4 = (
        self.current_position[0] - self.resource_half_square, self.current_position[1] + self.resource_half_square)

        true_v1 = (max(v1[0], 0), max(v1[1], 0))
        true_v2 = (min(v2[0], map.map_dim[0]), max(v2[1], 0))
        true_v3 = (min(v3[0], map.map_dim[0]), min(v3[1], map.map_dim[1]))
        true_v4 = (max(v4[0], 0), min(v4[1], map.map_dim[1]))

        # Now that we have the box, let's count how many resources are inside
        res_found = 0
        for i in range(true_v1[0], true_v2[0]):
            for j in range(true_v1[1], true_v4[1]):
                if map.resource_map[i][j] != 0:
                    res_found = res_found + 1

        # print("res_found " + str(res_found))
        return res_found

    def __str__(self) -> str:
        return "Particle: \n" + \
               "\tcurrent_position: " + str(self.current_position) + " \n" + \
               "\tvelocity: " + str(self.velocity) + " \n" + \
               "\tlocal_best: " + str(self.local_best) + " \n" + \
               "\tstarting_base: " + str(self.starting_base) + " \n" + \
               "\tresource_range : " + str(self.resource_half_square) + " \n"
