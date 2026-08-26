import cv2
import numpy as np
import matplotlib.pyplot as plt
import sampler.intersections as intersections
import itertools

import sampler.randompath as randompath
import plotutils
import rasterizer
np.random.seed(152)


    
N = 1000
offset = 30
scale = 10

lines = [
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], offset, scale),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)],  offset, scale),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)],  offset, scale),
 np.array([(100, 100), (180, 180)]),
 np.array([(101, 180), (181, 100)]),
]


img = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a white line
for pl, color in zip(lines, itertools.cycle(plotutils.COLORS)):
    rasterizer.draw_polyline_cv2(img, pl, plotutils.color_to_bgr(color))

img2 = np.zeros((500, 500, 3), dtype=np.uint8)

for pl, color in zip(lines, itertools.cycle(plotutils.COLORS)):
    px = rasterizer.get_polyline_pixels(pl, img2.shape[:2])
    img2[px[:, 0], px[:, 1]] = plotutils.color_to_bgr(color)

cv2.imwrite("debug_files/cv2lines.png", img)
cv2.imwrite("debug_files/skimagelines.png", img2)
