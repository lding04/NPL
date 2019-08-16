import requests
import json
import math


bd_ak = '0YAUctq0xrkh3Sc9u2YGEfgxHbYgzODt'
bd_coordinate_URL = 'http://api.map.baidu.com/geocoder/v2/'


def bd_geo_encode(address):
	res = requests.get(bd_coordinate_URL, params={'address': address, 'output': 'json', 'ak': bd_ak})
	try:
		coordinate_data = res.json()['result']['location']
	except KeyError:
		return '百度限制访问'
	else:
		return [coordinate_data['lng'], coordinate_data['lat']]


class Ruler:
	def __init__(self):
		self.pk = 180 / 3.14169

	def rule(self, pt1, pt2):
		# pt(longitude, latitude)
		a1 = pt1[1] / self.pk
		a2 = pt1[0] / self.pk
		b1 = pt2[1] / self.pk
		b2 = pt2[0] / self.pk
		t1 = math.cos(a1) * math.cos(a2) * math.cos(b1) * math.cos(b2)
		t2 = math.cos(a1) * math.sin(a2) * math.cos(b1) * math.sin(b2)
		t3 = math.sin(a1) * math.sin(b1)
		tmp = t1 + t2 + t3 if t1 + t2 + t3 < 1 else 1
		tt = math.acos(tmp)
		return 6366000 * tt
