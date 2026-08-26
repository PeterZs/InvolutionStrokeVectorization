import graph_extraction.image as image
import cv2
import numpy as np

img = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)
img =  (img > 0).astype(np.uint8) * 255
res = image.skeleton(img)
cv2.imwrite("skel.png", res)