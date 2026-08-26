import numpy as np
import cv2
import brushengine.brushengine as brushengine


w, h = 800, 500
canvas = np.zeros((h, w), np.uint8)

pos = 5.1, 5.2
patch = np.ones((50, 50), dtype=np.uint8)*255

patchr = brushengine.rotate_img(patch, 35)
cv2.imwrite("rotated.png", patchr)

newpatch, yslice, xslice = brushengine.get_subpixel_patch_pos(patch, pos, w, h)
canvas[yslice, xslice] = newpatch
cv2.imwrite("patch.png", newpatch)
cv2.imwrite("translated.png", canvas)
