import cv2
import numpy as np
from skimage.filters import sato, hessian, frangi, meijering
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt

def local_contrast_normalization(img, kernel_size=31):
    """
    Perform local min-max contrast normalization on a grayscale image.
    Each pixel is scaled so that local min -> 0 and local max -> 1.

    Args:
        img (ndarray): Grayscale image (uint8 or float)
        kernel_size (int): Neighborhood window size (odd number)

    Returns:
        norm_img (float32): Locally normalized image in [0,1]
    """
    img = img.astype(np.float32)
    
    # Local min and max via morphological filters
    local_min = cv2.erode(img, np.ones((kernel_size, kernel_size), np.uint8))
    local_max = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8))
    
    cv2.imwrite("localmin.png", (local_min).astype(np.uint8))
    cv2.imwrite("localmax.png", (local_max).astype(np.uint8))

    # Avoid divide-by-zero
    denom = (local_max - local_min)
    denom[denom < 1e-6] = 1.0
    
    norm = (img - local_min) / denom
    norm = np.clip(norm, 0, 1)
    return (norm*255).astype(np.uint8)

# --- Load grayscale image ---
img = cv2.imread("testcase/thin_blurred.png", cv2.IMREAD_GRAYSCALE)
 
img = 255-img
cv2.imwrite("inverted.png", img)

img = local_contrast_normalization(img, kernel_size=3)

cv2.imwrite("bettercontrast.png", img)
# Apply a binary threshold
# _, binary= cv2.threshold(binary, 100, 255, cv2.THRESH_BINARY)
# cv2.imwrite("bin.png", binary)