import numpy as np
import cv2


img = cv2.imread("testcases/input/1.png", cv2.IMREAD_GRAYSCALE)


_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
adjusted = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
 
adjusted = 255-adjusted
cv2.imwrite("debug_files/adaptivethresh.png", adjusted)
binary = cv2.subtract(binary, adjusted)
cv2.imwrite("debug_files/bin_adjusted.png", binary)
res = cv2.distanceTransform(255- binary, cv2.DIST_L2, 5)
dist_dbg= cv2.normalize( 
        res,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
cv2.imwrite("debug_files/dt.png", dist_dbg)


