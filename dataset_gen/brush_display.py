import numpy as np
import cv2
from svgpathtools import svg2paths2, wsvg, Path, Line, svgstr2paths
import os
import brushengine.brushengine as brushengine
import svgutils
import drawutils
import json
np.random.seed(25)



IN_PATH = 'debug_files/teststroke.svg'
TARGET_SIZE = 2000
with open(IN_PATH, 'r') as f:
    svg = f.read() 
svg = svgutils.normalize_svg(svg, target_size=TARGET_SIZE)
BRUSHES_TO_USE = ["calligraphy_marker_small", "pencil_2", "ink", "pencil_3", "small_pen", "watercolor_2", "watery_ink", "tiny_pen"]
BRUSHES_TO_USE = []

brushes = drawutils.get_all_brushes(BRUSHES_TO_USE)
for b in brushes:
    print("processing", b.name)
    print("max brush size:", brushengine.get_max_brush_size(b))
    print()
    paths, attributes = svgstr2paths(svg)
    points = []
    for p in paths:
        points.append(svgutils.getpoints(p, n=80))

    size = TARGET_SIZE
    img = np.ones((int(size), int(size), 4), dtype=np.uint8) * 255
    
    painted= brushengine.draw_patch_along_polyline(
        img, points[0], b.tips, b.params, texture=b.texture)
    cv2.imwrite(f"brush_display_results/{b.name}.png", painted)




