import numpy as np
from PIL import Image

resources = [
	"coal",
	"iron",
	"copper",
	"stone",
	"uranium",
]

resourceColours = {
	"coal": [0, 0, 0],
	"iron": [104, 132, 146],
	"copper": [203, 97, 53],
	"stone": [174, 154, 107],
	"uranium": [0, 177, 0]
}


def __highlight_resource(res_matrix, image, resName, on_map):
	searching = resourceColours[resName]
	# Scan the image looking for the color
	for i in range(0, image.shape[0]):
		for j in range(0, image.shape[1]):
			pixel = image[i][j]
			if pixel[0] == searching[0] and pixel[1] == searching[1] and pixel[2] == searching[2]:
				res_matrix[i][j] = on_map
				res_matrix[i][j] = on_map
				res_matrix[i][j] = on_map

	return res_matrix


def detect_resource(image_name) -> np.ndarray:
	"""
	Scan the input matrix looking for the different resources: coal, iron, copper and uranium.
	Foreach of them create a new image with everything black and only the relevant resource spots white (to ease the
	job of blob_detection). Then it runs blob_detection and save the result on a new matrix.
	Save the result matrix on file the first time, return it if it has already been computed.
	:param image_name: name of the image
	:return: output matrix
	"""

	global result

	# Retrieve the image and convert it to an array
	img = Image.open('data/examples/' + image_name)
	img_array = np.asarray(img, dtype="int32")
	image_path = 'data/cached_matrices/' + image_name.replace('.png', '') + '_resources.npy'

	try:
		result = np.load(image_path)
	except IOError:

		print('<error> Resource matrix file does not exist or cannot be read.')

		# Find all resources and put them on the matrix
		result = np.zeros((img_array.shape[0], img_array.shape[1]))
		for i in range(0, len(resources)):
			__highlight_resource(result, img_array, resources[i], i + 1)
			print('End processing of ' + resources[i])

		np.save(image_path, result)
		print('Resource processing completed!')
	finally:
		return result
