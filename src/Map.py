from src.detect_resources import detect_resource
from src.detect_water import detect_water


class Map:

	resource_map = None
	water_map = None
	map_dim = None

	def __init__(self, image_name):
		self.resource_map = detect_resource(image_name)
		self.water_map = detect_water(image_name)
		self.map_dim = self.resource_map.shape
