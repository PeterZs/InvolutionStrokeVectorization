import numpy as np
import cv2
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import generic_filter

def highlight_tangent_regions(image, max_d):
    # 1. Normalize locally to handle "relative intensity"
    # We use a high-pass filter or local normalization to remove global lighting
    img_float = image.astype(np.float32) / 255.0
    
    # 2. Detect Ridge-like structures using Hessian Eigenvalues
    # sigma determines the width of the lines you are looking for
    H_elems = hessian_matrix(img_float, sigma=0.2, order='rc')
    e1, e2 = hessian_matrix_eigvals(H_elems)
    
    # e2 represents the principal curvature. 
    # Large absolute values of e2 indicate a ridge (line/curve).
    # We take the absolute to find both bright lines on dark and vice-versa.
    ridges = e2
    ridges[ridges>0] =0
    
    ridges_dbg= cv2.normalize(
        ridges,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    cv2.imwrite("debug_files/ridges.png", ridges_dbg)

    # 3. Non-Maximum Suppression (NMS)
    # We need to find the "spine" of the lines to calculate distance accurately.
    # Instead of thresholding, we find pixels that are local maxima in their neighborhood.
    def is_local_max(buffer):
        central_pixel = buffer[len(buffer) // 2]
        return 1 if central_pixel == np.max(buffer) and central_pixel > 0.01 else 0

    # This finds the "skeleton" of the ridges based on relative local intensity
    ridge_spine = generic_filter(ridges, is_local_max, size=3).astype(np.uint8)
    
    spine_dbg = (ridge_spine*255).astype(np.uint8)
    cv2.imwrite("debug_files/spine.png", spine_dbg)
    # 4. Distance Transform
    # Calculates distance from every pixel to the nearest ridge spine
    dist_map, labels = cv2.distanceTransformWithLabels((1 - ridge_spine), cv2.DIST_L2, 5)

    # 5. Finding "Close" Ridges
    # We want to find areas where two DIFFERENT ridge spines are within distance d
    # A simple way: find areas where the distance to the nearest ridge is <= d/2
    # because the "midpoint" between two lines distance d apart is d/2 away from both.
    
    potential_regions = (dist_map <= max_d / 2) & (dist_map > 0)
    
    # 6. Refine: Highlight only if the region is sandwiched between ridges
    # We look for pixels where the distance transform has a local maximum 
    # (the "valley" between two ridges)
    is_midpoint = generic_filter(dist_map, is_local_max, size=3)
    
    # Final Result: Midpoints between ridges that are at most distance d apart
    highlight = np.where((is_midpoint > 0) & (dist_map <= max_d / 2), 255, 0).astype(np.uint8)
    
    return highlight, ridges

# --- Usage ---
image = cv2.imread("testcases/input/10.png", cv2.IMREAD_GRAYSCALE)
d_pixels = 8  # Set your maximum distance
result, ridge_map = highlight_tangent_regions(image, d_pixels)

cv2.imwrite("debug_files/tangents.png", result)