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

import json
import os.path
import cv2
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
            
            maskfilepath = filename[:-4] + "_mask.png"
            mask = np.zeros((M,N))
            
            for i in range(M):
                for j in range(N):
                    for shape in shapes:
                        if self.point_in_poly(j,i,shape['points']):
                            mask[i][j] = int(shape['label'])
            
            print np.unique(mask)
            # mask = cv2.flip(mask, 0)

            cv2.imwrite(str(maskfilepath), mask*20)

        except Exception, e:
            raise LabelFileError(e)

    def point_in_poly(self, x, y, poly):
        # point in polygon algorithm
        # Ray casting method
		n = len(poly)
		inside = False

		p1x,p1y = poly[0]
		for i in range(n+1):
			p2x,p2y = poly[i % n]
			if y > min(p1y,p2y):
				if y <= max(p1y,p2y):
					if x <= max(p1x,p2x):
						if p1y != p2y:
							xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
						if p1x == p2x or x <= xints:
							inside = not inside
			p1x,p1y = p2x,p2y

		return inside

    @staticmethod
    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix

