import numpy as np


def detect_water(data):
	global result
	try:
		result = np.load('data/cached_matrices/water.npy')
	except IOError:

		print('<error> Water matrix file does not exist or cannot be read.')

		# Create matrix for the result
		rows = data.shape[0]
		columns = data.shape[1]
		result = np.zeros((rows, columns))

		# Scan image looking for the colors
		for r in range(0, rows):
			for c in range(0, columns):
				if (data[r][c][0] == 38 and data[r][c][1] == 64 and data[r][c][2] == 73) or (
					data[r][c][0] == 51 and data[r][c][1] == 83 and data[r][c][2] == 95):
					result[r][c] = 1

		np.save('data/cached_matrices/water', result)
		print('Water processing completed!')
	finally:
		return result
