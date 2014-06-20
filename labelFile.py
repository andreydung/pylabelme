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
import cv2
import json
import os.path
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from base64 import b64encode, b64decode

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

	def save(self, filename, shapes, imagePath, imageData,
			lineColor=None, fillColor=None):
		try:
			with open(filename, 'wb') as f:
				json.dump(dict(
					shapes=shapes,
					lineColor=lineColor, fillColor=fillColor,
					imagePath=imagePath,
					imageData=b64encode(imageData)),
					f, ensure_ascii=True, indent=2)

			# Also export the mask image
			shape = QImage.fromData(imageData).size()
			M = shape.height()
			N = shape.width()
			
			fullmask = np.zeros((M,N))

			colors = np.array([[255,255,255],
                   [0, 255, 0],
                   [0, 0, 255],
                   [255, 0, 0]
                  ])
			
			for shape in shapes:
				mask = self.polygon(M, N, shape['points'])
				fullmask[mask] = int(shape['label'])

			fullmask = fullmask.astype(int)

			maskfilepath = filename[:-4] + "_truth.csv"
			masktestfilepath = filename[:-4] + "_mask.png"

			np.savetxt(str(maskfilepath), fullmask, fmt = "%d")
			cv2.imwrite(str(masktestfilepath), colors[fullmask])

		except Exception, e:
			raise LabelFileError(e)

	def polygon(self, M, N, poly):
		"""
		Return a mask matrix
		Where points inside the polygon is 1, outside is 0
		"""
		out = np.zeros((M, N)). astype(bool)

		n = len(poly)
		for i in range(M):
			intersection_x = i
			intersection_y = []

			# check through all edges
			for edge in range(n + 1):
				v1_x, v1_y = poly[edge % n]
				v2_x, v2_y = poly[(edge + 1) % n]

				A1, B1, C1 = self.getABC(v1_x, v1_y, v2_x, v2_y)
				A2 = 1
				B2 = 0
				C2 = i

				# find intersection
				det = A1*B2 - A2*B1
				if (det != 0):
					tmp = (A1 * C2 - A2 * C1)/det
					if tmp >= min(v1_y, v2_y) and tmp <= max(v1_y, v2_y):
						intersection_y.append(int(tmp))

			intersection_y = sorted(list(set(intersection_y)))

			if len(intersection_y) > 1:
				for k in range(1, len(intersection_y), 2):
					out[intersection_x, intersection_y[k - 1]:intersection_y[k]] = True

		return out

	def getABC(self, x1, y1, x2, y2):
		A = y2 - y1
		B = x1 - x2
		C = A*x1 + B*y1
		return (A, B, C)

	@staticmethod
	def isLabelFile(filename):
		return os.path.splitext(filename)[1].lower() == LabelFile.suffix

