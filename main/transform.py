import numpy as np
def compute_matrix_for_rescaling(in_w: float, in_h: float, target: float, align: str = "center") -> np.ndarray:
    """
    Compute matrix (a, b, c, d, e, f) for uniform scaling to fit (no stretching)
    into a square of size `target`. `align` can be "center" or "top-left".
    - a = scale_x = scale
    - d = scale_y = scale
    - b = c = 0
    - e = tx, f = ty (translation)
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input width and height must be positive numbers")

    s = min(target / in_w, target / in_h)
    scaled_w = in_w * s
    scaled_h = in_h * s

    if align == "center":
        tx = (target - scaled_w) / 2.0
        ty = (target - scaled_h) / 2.0
    elif align == "top-left":
        tx = 0.0
        ty = 0.0
    else:
        raise ValueError("invalid align")

    # matrix(a b c d e f) with no skew/rotation
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def apply_transform(matrix: np.ndarray, points: np.ndarray):
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack([points, ones])
    transformed = hom @ matrix.T
    return transformed[:, :2]