#
# Copyright (C) 2011 Michael Pitidis, Hussein Abdulwahid.
#
# This file is part of Labelme.
#
# Labelme is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Labelme is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Labelme.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import division
import cv2
import json
import os.path
import numpy as np
import math

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from base64 import b64encode, b64decode
import base64

def polygon(M, N, poly):
	"""
	Return a mask matrix
	Where points inside the polygon is 1, outside is 0
	"""
	out = np.zeros((M, N)). astype(bool)

	n = len(poly)
	for i in range(N):
		
		# horizontal scanning
		intersection_x = i
		intersection_y = []

		# check through all edges
		for edge in range(n):
			v1_x, v1_y = poly[edge % n]
			v2_x, v2_y = poly[(edge + 1) % n]

			v1_x = int(v1_x)
			v1_y = int(v1_y)
			v2_x = int(v2_x)
			v2_y = int(v2_y)

			# assert (v1_x <= M and v2_x <= M and v1_y <= N and v2_y <= N)

			A1, B1, C1 = getABC(v1_x, v1_y, v2_x, v2_y)
			A2 = 1
			B2 = 0
			C2 = i

			# find intersection
			if intersection_x > min(v1_x, v2_x) and intersection_x <= max(v1_x, v2_x):
				det = A1*B2 - A2*B1
				if (det != 0):
					tmp = int((A1 * C2 - A2 * C1)/det)
					intersection_y.append(tmp)

		intersection_y = sorted(intersection_y)
		if len(intersection_y) > 1:
			for k in range(1, len(intersection_y), 2):
				out[intersection_y[k - 1]:intersection_y[k], intersection_x] = True
	return out

def getABC(x1, y1, x2, y2):
	A = y2 - y1
	B = x1 - x2
	C = A*x1 + B*y1
	return (A, B, C)


class LabelFileError(Exception):
	pass

class LabelFile(object):
	suffix = '.lif'

	def __init__(self, filename=None):
		self.shapes = ()
		self.imagePath = None
		self.imageData = None
		if filename is not None:
			self.load(filename)

	def load(self, filename):
		try:
			with open(filename, 'rb') as f:
				data = json.load(f)
				imagePath = data['imagePath']
				imageData = b64decode(data['imageData'])
				imageData2 = b64decode(data['imageData2'])
				lineColor = data['lineColor']
				fillColor = data['fillColor']
				shapes = ((s['label'], s['points'], s['line_color'], s['fill_color'])\
						for s in data['shapes'])
				# Only replace data after everything is loaded.
				self.shapes = shapes
				self.imagePath = imagePath
				self.imageData = imageData
				self.lineColor = lineColor
				self.fillColor = fillColor
		except Exception, e:
			raise LabelFileError(e)

	def save(self, filename, shapes, imagePath, imageData, imageData2,
			lineColor=None, fillColor=None):
		try:
			with open(filename, 'wb') as f:
				json.dump(dict(
					shapes=shapes,
					lineColor=lineColor, fillColor=fillColor,
					imagePath=imagePath,
					imageData=b64encode(imageData),
					imageData2=b64encode(imageData2)),
					f, ensure_ascii=True, indent=2)

			# Also export the mask image
			shape = QImage.fromData(imageData).size()
			M = shape.height()
			N = shape.width()
			
			fullmask = np.zeros((M,N)).astype(int)

			# Lookup table
			colors = np.array([[0,128,128],
							   [0, 0, 128],
							   [0, 128, 0]])
			
			# combine mask of different labels
			for shape in shapes:
				mask = polygon(M, N, shape['points'])
				fullmask[mask] = int(shape['label'])
				print shape['label']

			maskfilepath = filename[:-4] + "_truth.csv"
			masktestfilepath = filename[:-4] + "_mask.png"

			np.savetxt(str(maskfilepath), fullmask, fmt = "%d")
			cv2.imwrite(str(masktestfilepath), colors[fullmask])

		except Exception, e:
			raise LabelFileError(e)

	@staticmethod
	def isLabelFile(filename):
		return os.path.splitext(filename)[1].lower() == LabelFile.suffix

