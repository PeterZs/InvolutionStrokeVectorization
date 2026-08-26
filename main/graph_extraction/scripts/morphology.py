import cv2
import numpy as np
from skimage.morphology import skeletonize

# Load grayscale image
image_path = 'testcase/thin.png'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Invert image
img_inverted = 255 - img

# Threshold: make all nonzero pixels 1
_, img_binary = cv2.threshold(img_inverted, 0, 1, cv2.THRESH_BINARY)

# Skeletonize (requires boolean image)
img_skeleton = skeletonize(img_binary.astype(bool))

# Convert skeleton back to uint8 for OpenCV operations
img_skeleton_uint8 = (img_skeleton.astype(np.uint8)) * 255

# Dilate by 1 pixel
kernel = np.ones((2, 2), np.uint8)  # 3x3 kernel
img_skeleton_uint8= cv2.dilate(img_skeleton_uint8, kernel, iterations=1)

#invert back
img = 255-img_skeleton_uint8
# Save result
cv2.imwrite('testcase/morphed.png', img)

print("Processing complete. Saved as 'output_image.png'.")