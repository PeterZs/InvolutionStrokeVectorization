import numpy as np
import cv2

def circ(radius):
    size=2*radius+1
    kernel = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel

def remove_median(img, debug=False):
    blurred = cv2.medianBlur(img, 5)  # 5x5 kernel
    img =  cv2.subtract(img, blurred)
    
    if debug:
        cv2.imwrite("debug_files/medianremove.png", img)
        
    
    _, binary= cv2.threshold(img, 1 , 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imwrite("debug_files/bin.png", binary)
    return binary
img = cv2.imread("testcases/input/10.png", cv2.IMREAD_GRAYSCALE)


_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
adjusted = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
 
adjusted = 255-adjusted
cv2.imwrite("debug_files/adaptivethresh.png", adjusted)
binary_before = binary.copy()
binary = cv2.subtract(binary, adjusted)
cv2.imwrite("debug_files/bin_adjusted.png", binary)
diff = cv2.subtract(binary_before, binary)
cv2.imwrite("debug_files/diff_adjusted.png", diff)



