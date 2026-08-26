import numpy as np
import cv2
import brushengine.brushengine as brushengine
from pprint import pprint

pos = 140.1, 160.2
scale = 1
angle = 0

w, h = 800, 500

canvas = np.ones((h, w, 4), np.uint8)*255

params, patch, texture = brushengine.load_brush("brushes/ink")
scale = params.size_pixels / patch.shape[0]

# patch = np.ones((50, 50, 1), dtype=np.uint8) * 255

res, yslice, xslice = brushengine.patch_transform(patch, w, h, pos, angle, scale)
colorstack = np.array([0, 0, 0], dtype=np.uint8).reshape(1, 1, 3)
colorpatch = np.tile(colorstack, (*res.shape[:2], 1))
print(colorpatch.shape)
tmp = brushengine.add_alpha_to_color(colorpatch, res.astype(np.uint8))
cv2.imwrite("patch_colored.png", tmp)
tmp2 = canvas[yslice, xslice]
canvas[yslice, xslice] = brushengine.alpha_blend_4channel(tmp, tmp2)
cv2.imwrite("newpatch.png", res)
cv2.imwrite("canvas.png", canvas)

