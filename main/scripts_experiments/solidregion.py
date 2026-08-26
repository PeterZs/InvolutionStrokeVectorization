import cv2
import numpy as np
import graph_extraction.image as gimg

path = "debug_files/balloon/balloon_centerline.png"
# path = "debug_files/hat/hat_centerline.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
binary = (img > 0).astype(np.uint8) * 255

# 3x3 with one corner forbidden, plus its 3 rotations.
corner_cut = np.array([
    [1, 1, -1],
    [1, 1,  1],
    [1, 1,  1],
], dtype=np.int8)

all_ones = np.ones((3, 3), dtype=np.int8)

kernels = [np.rot90(corner_cut, r) for r in range(4)] + [all_ones]

union = np.zeros_like(binary)
for k in kernels:
    mask, _ = gimg.detect_and_mark(binary, k)
    union = np.maximum(union, mask)

cv2.imwrite("solidregion_union.png", union)
print(f"saved solidregion_union.png, marked pixels: {(union > 0).sum()}")
