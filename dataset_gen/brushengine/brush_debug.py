import numpy as np
import cv2
from svgpathtools import svg2paths2, wsvg, Path, Line, svgstr2paths
import os
import brushengine
np.random.seed(25)





brushfolder = 'brushes'
brush = brushengine.load_brush("brushengine/brushes/small_pen")
# texture = None
# params.distance_fraction = 10
# params.texture_mode = 0

img = np.ones((int(1200), int(1920), 4), dtype=np.uint8) * 255
print("launching paint op")

points = np.array([[-200, 200.5], [800, 200.5]])
painted= brushengine.draw_patch_along_polyline(
    img, points, brush.tips, brush.params, texture=brush.texture)
print("done")

points = np.array([[200, 400.5], [800, 400.5]])
painted= brushengine.draw_patch_along_polyline(
    painted, points, brush.tips, brush.params, texture=brush.texture)


cv2.imwrite("debug_files/painted.png", painted)


