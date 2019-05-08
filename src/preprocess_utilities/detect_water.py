import numpy as np
from PIL import Image


def detect_water(image_name):
	"""
	Scan the input matrix and detect where there is water on the map.
	Save the result matrix on file the first time, return it if it has already been computed.
	:param image_name: name of the image
	:return: output binary matrix (0 = earth, 1 = water)
	"""

	global result

	# Retrieve the image and convert it to an array
	img = Image.open("data/examples/" + image_name)
	img_array = np.asarray(img, dtype="int32")
	image_path = 'data/cached_matrices/' + image_name.replace('.png', '') + '_water.npy'

	try:
		result = np.load(image_path)
	except IOError:

		print('<error> Water matrix file does not exist or cannot be read.')

		# Create matrix for the result
		rows = img_array.shape[0]
		columns = img_array.shape[1]
		result = np.zeros((rows, columns))

		# Scan image looking for the colors
		for r in range(0, rows):
			for c in range(0, columns):
				if (img_array[r][c][0] == 38 and img_array[r][c][1] == 64 and img_array[r][c][2] == 73) or (
					img_array[r][c][0] == 51 and img_array[r][c][1] == 83 and img_array[r][c][2] == 95):
					result[r][c] = 1

		np.save(image_path, result)
		print('Water processing completed!')
	finally:
		return result
