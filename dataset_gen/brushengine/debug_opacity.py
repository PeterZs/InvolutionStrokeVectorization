import numpy as np
import cv2
import brushengine.brushengine as brushengine
from pprint import pprint

pos = 140.1, 160.2
scale = 1
angle = 0

w, h = 50, 50

canvas = np.ones((h, w, 4), np.uint8)*255
canvas2 = np.ones((h, w, 4), np.uint8)*255
canvas2[..., -1] *=0
res = brushengine.alpha_blend_4channel(canvas2, canvas)


cv2.imwrite("blended.png", res)

