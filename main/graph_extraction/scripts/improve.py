import cv2
import matplotlib.pyplot as plt
print("read")
import numpy as np


from scipy.ndimage import maximum_filter, rank_filter, minimum_filter

def circ(radius):
    size=2*radius+1
    kernel = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel

img = cv2.imread("testcases/input/1.png", cv2.IMREAD_GRAYSCALE)
kernel = cv2.imread("debug_files/conv2.png", cv2.IMREAD_GRAYSCALE)
kernel = kernel.astype(np.float32)
kernel -= 127
kernel = kernel/kernel.sum()

result = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

cv2.imwrite("debug_files/convolved.png", result)

# adjusted = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1 )

# adjusted = 255-adjusted
# cv2.imwrite("adjusted.png", adjusted)

# img[img < 10] = 0
# resmask = np.zeros_like(img)
# for i in range(1):
#     modified = img.copy()
#     modified[modified==0]= 255
#     rk = modified <= rank_filter(modified, 3, (5, 5))
#     notblack = img > 0
#     hasblack = minimum_filter(img, 5) == 0
#     mask = rk & notblack & hasblack
#     img = cv2.subtract(img, np.uint8(mask*255))
#     cv2.imwrite(f"improved_{i}.png", img)
#     cv2.imwrite(f"mask_{i}.png", np.uint8(mask*255))
#     cv2.imwrite(f"modified{i}.png", modified)
#     cv2.imwrite(f"rankmask_{i}.png", rank_filter(modified, 3, size=(5,5)))





