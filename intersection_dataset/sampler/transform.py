import numpy as np
import re
from math import radians, cos, sin, tan

def rotate_mat(angle, center):
    cos_a, sin_a = cos(angle), sin(angle)
    cx, cy = center
    return np.array([
                        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
                        [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy],
                        [0, 0, 1]
                    ], dtype=float)


def apply_transform(matrix: np.ndarray, points: np.ndarray):
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack([points, ones])
    transformed = hom @ matrix.T
    return transformed[:, :2]

def apply_transform_lines(lines: list[np.ndarray], mat):
    return[apply_transform(mat, pl) for pl in lines]

class AffineTransform:
    def __init__(self, matrix: np.ndarray | None = None):
        """Initialize with a 3x3 matrix (defaults to identity)."""
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = np.array(matrix, dtype=float).reshape(3, 3)

    @staticmethod
    def from_svg_matrix2(matrix_str: str) -> "AffineTransform":
        """
        Parse an SVG transform string like:
        'matrix(a,b,c,d,e,f)' -> AffineTransform
        """
        matrix_str = matrix_str.strip()
        match = re.match(r"matrix\(([^)]+)\)", matrix_str)
        if not match:
            raise ValueError(f"Invalid matrix string: {matrix_str}")
        values = [float(x) for x in re.split(r"[,\s]+", match.group(1).strip())]
        if len(values) != 6:
            raise ValueError(f"Matrix requires 6 values: {matrix_str}")

        a, b, c, d, e, f = values
        mat = np.array([
            [a, c, e],
            [b, d, f],
            [0, 0, 1]
        ], dtype=float)
        return AffineTransform(mat)
    
    @staticmethod
    def from_svg_matrix(transform_str: str) -> "AffineTransform":
        """
        Parse SVG transform strings such as:
        'matrix(a,b,c,d,e,f)',
        'translate(tx, ty)',
        'scale(sx, sy)',
        'rotate(angle[, cx, cy])',
        'skewX(angle)',
        'skewY(angle)',
        or combinations like:
        'translate(508.3,110) scale(0.74)'
        """
        transform_str = transform_str.strip()
        if not transform_str:
            return AffineTransform()

        # Pattern to match transform commands
        pattern = re.compile(r"(\w+)\(([^)]+)\)")
        transforms = pattern.findall(transform_str)

        result = AffineTransform().matrix

        for name, args_str in transforms:
            args = [float(a) for a in re.split(r"[,\s]+", args_str.strip()) if a]
            name = name.lower()

            if name == "matrix":
                if len(args) != 6:
                    raise ValueError(f"matrix() expects 6 arguments, got {len(args)}")
                a, b, c, d, e, f = args
                mat = np.array([
                    [a, c, e],
                    [b, d, f],
                    [0, 0, 1]
                ], dtype=float)

            elif name == "translate":
                tx = args[0]
                ty = args[1] if len(args) > 1 else 0.0
                mat = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ], dtype=float)

            elif name == "scale":
                sx = args[0]
                sy = args[1] if len(args) > 1 else sx
                mat = np.array([
                    [sx, 0, 0],
                    [0, sy, 0],
                    [0, 0, 1]
                ], dtype=float)

            elif name == "rotate":
                angle = radians(args[0])
                cos_a, sin_a = cos(angle), sin(angle)
                if len(args) == 3:
                    cx, cy = args[1], args[2]
                    # Rotation about a point (cx, cy)
                    mat = np.array([
                        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
                        [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy],
                        [0, 0, 1]
                    ], dtype=float)
                else:
                    mat = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a,  cos_a, 0],
                        [0, 0, 1]
                    ], dtype=float)

            elif name == "skewx":
                angle = radians(args[0])
                mat = np.array([
                    [1, tan(angle), 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ], dtype=float)

            elif name == "skewy":
                angle = radians(args[0])
                mat = np.array([
                    [1, 0, 0],
                    [tan(angle), 1, 0],
                    [0, 0, 1]
                ], dtype=float)

            else:
                raise ValueError(f"Unsupported transform: {name}")

            result = result @ mat

        return AffineTransform(result)

    def combine(self, other: "AffineTransform") -> "AffineTransform":
        """
        Combine with another transform (apply 'other' after this one).
        Returns a new AffineTransform.
        """
        new_matrix = self.matrix @ other.matrix
        return AffineTransform(new_matrix)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the transform to an (n,2) array of points.
        Returns transformed (n,2) array.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be of shape (n, 2)")

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        hom = np.hstack([points, ones])
        transformed = hom @ self.matrix.T
        return transformed[:, :2]

    def __repr__(self):
        return f"AffineTransform(\n{self.matrix}\n)"