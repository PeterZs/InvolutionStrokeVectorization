from xml.etree import ElementTree as ET
import re
import os
import io
import numpy as np
from svgpathtools import Line, CubicBezier, QuadraticBezier, Path, parse_path
from bs4 import BeautifulSoup
from math import radians, cos, sin, tan
from PIL import Image
import cairosvg
from typing import Callable


def save_webp_from_svg(svgstr, output_path, grayscale=True, out_format = 'WEBP'):
    png_bytes = cairosvg.svg2png(bytestring=svgstr.encode('utf-8'), background_color='white')

    image = Image.open(io.BytesIO(png_bytes))
    if grayscale:
        # Convert the image to grayscale
        image = image.convert('L')
    image.save(output_path, format=out_format, lossless=True)

def rm_consecutive(arr: np.ndarray):
    diff_mask = np.ones(len(arr), dtype=bool)
    diff_mask[1:] = np.any(arr[1:] != arr[:-1], axis=1)
    return arr[diff_mask]

def getpoints(path: Path, n=80):
    assert len(path) > 0
    t = np.linspace(0, 1, n)
    p = []
    for seg in path:
        complex_points = seg.points(t)
        points = np.column_stack((complex_points.real, complex_points.imag))
        p.append(points)

    return rm_consecutive(np.concat(p))


def getpoints_maxdist(path: Path, n: int = 80, d: float = 0.1):
    """Same as `getpoints` for curve segments; for Line segments, sample just
    enough points (>=2) so adjacent samples are at most `d` apart, avoiding
    overcrowding on short lines.
    """
    assert len(path) > 0
    t_curve = np.linspace(0, 1, n)
    p = []
    for seg in path:
        if isinstance(seg, Line):
            n_line = max(2, int(np.ceil(seg.length() / d)) + 1)
            t = np.linspace(0, 1, n_line)
        else:
            t = t_curve
        complex_points = seg.points(t)
        points = np.column_stack((complex_points.real, complex_points.imag))
        p.append(points)

    return rm_consecutive(np.concat(p))

def points_to_path(points: np.ndarray, close: bool = False) -> Path:
    """
    Convert an (n, 2) numpy array of 2D points into an svgpathtools.Path
    composed of Line segments between consecutive points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) where each row is (x, y).
    close : bool, optional
        If True, add a final Line from the last point back to the first point
        (default False).

    Returns
    -------
    svgpathtools.Path
        A Path consisting of Line segments connecting the points in order.

    Raises
    ------
    ValueError
        If `points` does not have shape (n, 2) or if n < 2.
    """
    # Validate input
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a 2D array with shape (n, 2).")
    n = pts.shape[0]
    if n < 2:
        pts = np.vstack([pts, pts])
        # raise ValueError("At least two points are required to form a path.")

    # Convert to complex numbers: x + y*j
    complex_pts = (pts[:, 0] + 1j * pts[:, 1]).tolist()

    # Build Line segments between consecutive points
    segments = []
    for i in range(n - 1):
        segments.append(Line(complex_pts[i], complex_pts[i + 1]))

    # Optionally close the path
    if close:
        segments.append(Line(complex_pts[-1], complex_pts[0]))
    return Path(*segments)


def get_overall_min_max(arrays):
    """
    Takes a list of np.ndarray of shape (n, 2) and returns the overall
    min and max for x and y dimensions across all arrays.

    Parameters:
    arrays (list of np.ndarray): List where each element is an (n, 2) array.

    Returns:
    dict: A dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'
    """
    # Concatenate all arrays into a single array
    concatenated = np.vstack(arrays)

    # Compute min and max for each column (x: col 0, y: col 1)
    x_min, y_min = np.min(concatenated, axis=0)
    x_max, y_max = np.max(concatenated, axis=0)

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }

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




def remove_clip_paths(svg_string: str) -> str:
    """
    Remove all clip-path attributes and <clipPath> elements from an SVG string.

    Args:
        svg_string (str): The input SVG content as a string.

    Returns:
        str: The SVG string with clip paths removed.
    """
    # Parse the SVG string
    root = ET.fromstring(svg_string)
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Remove clip-path attributes on all elements
    for elem in root.findall(".//*[@clip-path]"):
        del elem.attrib['clip-path']

    # Remove all <clipPath> elements inside <defs>
    for defs in root.findall(".//svg:defs", ns):
        for clip in defs.findall("svg:clipPath", ns):
            defs.remove(clip)


    # Convert back to string
    svg_output = ET.tostring(root, encoding='unicode')

    return svg_output

def remove_duplicated(svg_string: str) -> str:

    #this function is used for the PimpMyDrawingDataset.
    #The first groups are empty or contain the background path.
    #Sometimes, this path is duplicated for some reason and this function removes these.
    root = ET.fromstring(svg_string)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    # all  <g> elements that are direct children of <svg>
    outer_gs = [elem for elem in root.findall("svg:g", ns)]

    for g in outer_gs[:-1]:
        # Only consider direct <path> children (not nested)
        paths = g.findall("svg:path", ns)

        if len(paths) > 1:
            # Remove all but the first
            for p in paths[1:]:
                g.remove(p)
            break
    svg_output = ET.tostring(root, encoding='unicode')

    # svg_output = re.sub(r'\bsvg:', '', svg_output)
    return svg_output



def _parse_length(value: str) -> float | None:
    """Parse SVG length (e.g., '100', '200px', '50%', '10cm') → float or None"""
    if not value:
        return None
    value = value.strip().lower()
    if value.endswith("px"):
        return float(value[:-2])
    if value.endswith("%"):
        # Percentages are relative; we’ll treat them as None (handled via viewBox)
        return None
    # Try to parse a bare number
    match = re.match(r"^[0-9.]+$", value)
    if match:
        return float(value)
    return None

#returns width, height
#should parse the svg width and height. Make sure to consider the viewbox. Should handle the case where width, height are in percent.
def get_svg_dimensions(svg: str) -> tuple[int, int]:
    soup = BeautifulSoup(svg, "xml")
    svg_tag = soup.find("svg")

    if not svg_tag:
        raise ValueError("invalid svg")

    width = _parse_length(str(svg_tag.get("width")))
    height = _parse_length(str(svg_tag.get("height")))

    # Parse viewBox: "minX minY width height"
    viewBox = svg_tag.get("viewBox")
    vb_width = vb_height = None
    if viewBox:
        parts = re.split(r"[\s,]+", viewBox.strip())
        if len(parts) == 4:
            try:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
            except ValueError:
                pass

    if vb_width is not None:
        width = vb_width
    if vb_height is not None:
        height = vb_height

    return (int(width or 0), int(height or 0))



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


def compute_matrix_for_rescaling_rect(in_w: float, in_h: float, target_w: float, target_h: float, align: str = "center") -> np.ndarray:
    """
    Uniform scaling to fit into a rectangle of size target_w x target_h.
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input width and height must be positive numbers")

    s = min(target_w / in_w, target_h / in_h)
    scaled_w = in_w * s
    scaled_h = in_h * s

    if align == "center":
        tx = (target_w - scaled_w) / 2.0
        ty = (target_h - scaled_h) / 2.0
    elif align == "top-left":
        tx = 0.0
        ty = 0.0
    else:
        raise ValueError("invalid align")

    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def _polyline_points_to_d(points_str: str) -> str | None:
    """Convert a polyline points attribute string to an SVG path d string."""
    coords = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", points_str)
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    coords = [float(c) for c in coords]
    pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    d = f"M {pts[0][0]},{pts[0][1]}"
    for x, y in pts[1:]:
        d += f" L {x},{y}"
    return d


#given an svg string, returns a list of all paths with:
# - a list of all its parent transform matrix strings (ordered from root to leaf)
# - the path string
def get_paths_with_transform(svg: str) -> list[tuple[list[str], str]]:
    soup = BeautifulSoup(svg, "lxml-xml")

    result = []

    for elem in soup.find_all(["path", "polyline"]):
        # Collect transform attributes from all ancestors (root -> elem)
        transforms = []
        if elem.has_attr("transform"):
            transforms.append(elem["transform"])
        parent = elem.parent
        while parent is not None and parent.name is not None:
            if parent.has_attr("transform"):
                transforms.append(parent["transform"])
            parent = parent.parent
        transforms.reverse()

        if elem.name == "path":
            d = elem.get("d")
        else:  # polyline
            d = _polyline_points_to_d(elem.get("points", ""))

        if d:
            result.append((transforms, d))

    return result


def transform_path(path: Path, func) -> Path:
    new_segments = []

    for segment in path:
        if isinstance(segment, Line):
            # Convert complex points to real (x, y)
            pts = np.array([[p.real, p.imag] for p in [segment.start, segment.end]], dtype=float)
            transformed_pts = func(pts)
            # Convert back to complex for svgpathtools
            new_segment = Line(transformed_pts[0,0] + 1j*transformed_pts[0,1],
                               transformed_pts[1,0] + 1j*transformed_pts[1,1])
            new_segments.append(new_segment)

        elif isinstance(segment, CubicBezier):
            pts = np.array([[p.real, p.imag] for p in [segment.start, segment.control1, segment.control2, segment.end]], dtype=float)
            transformed_pts = func(pts)
            new_segment = CubicBezier(transformed_pts[0,0] + 1j*transformed_pts[0,1],
                                      transformed_pts[1,0] + 1j*transformed_pts[1,1],
                                      transformed_pts[2,0] + 1j*transformed_pts[2,1],
                                      transformed_pts[3,0] + 1j*transformed_pts[3,1])
            new_segments.append(new_segment)
        elif isinstance(segment, QuadraticBezier):
            pts = np.array([[p.real, p.imag] for p in [segment.start, segment.control, segment.end]], dtype=float)
            transformed_pts = func(pts)
            new_segment = QuadraticBezier(transformed_pts[0,0] + 1j*transformed_pts[0,1],
                                      transformed_pts[1,0] + 1j*transformed_pts[1,1],
                                      transformed_pts[2,0] + 1j*transformed_pts[2,1]
                                      )
            new_segments.append(new_segment)
        else:
            raise NotImplementedError(f"Segment type {type(segment)} not supported")

    return Path(*new_segments)

def normalize_path_strings(path_data: tuple[list[str], str], root_transform: AffineTransform) -> str | None:

    mats, pstring = path_data
    total_transform = root_transform
    for i in mats:
        total_transform = total_transform.combine(AffineTransform.from_svg_matrix(i))
    try:
        path : Path = parse_path(pstring)
    except ValueError:
        return None

    def func(points: np.ndarray):
        return total_transform.apply(points)
    return transform_path(path, func).d()

def normalize_path(path_data: tuple[list[str], str], root_transform: AffineTransform) -> Path | None:

    mats, pstring = path_data
    total_transform = root_transform
    for i in mats:
        total_transform = total_transform.combine(AffineTransform.from_svg_matrix(i))
    try:
        path : Path = parse_path(pstring)
    except ValueError:
        return None

    def func(points: np.ndarray):
        return total_transform.apply(points)
    return transform_path(path, func)

def style_dict_to_string(style_dict):
    """Convert a dictionary back to a style string"""
    return "; ".join(f"{k}: {v}" for k, v in style_dict.items())

def parse_style(style_str):
    """Convert a style string into a dictionary"""
    style_dict = {}
    if style_str:
        for item in style_str.split(";"):
            if item.strip():
                key, value = item.split(":")
                style_dict[key.strip()] = value.strip()
    return style_dict

def get_svg_string(path_strings: list[str], target_w: int, target_h: int, points: list[tuple[float, float]] | None = None, point_radius: float = 3, pointcolor="black"):
    result= f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {target_w}, {target_h}"
    version="1.1">'''.strip()
    suffix = "</svg>"
    style_dict = {"fill": "none", "stroke": "black", "stroke-width": "0.3"}
    result += "\n"
    for p in path_strings:
        result +=  '<path d="' + p + '" style="' + style_dict_to_string(style_dict) +'"/>\n'
    if points is not None:
        for x, y in points:
            result += f'<circle cx="{x}" cy="{y}" r="{point_radius}" fill="{pointcolor}"/>\n'
    return result + suffix

def get_svg_string_paths(paths: list[Path], target_w: int, target_h: int, **kwargs):
    s = [p.d() for p in paths]
    return get_svg_string(s, target_w, target_h, **kwargs)



def normalize_svg(svg: str, target_size: int):

    width, height = get_svg_dimensions(svg)
    raw_paths = get_paths_with_transform(svg)



    root_transform= AffineTransform()
    if target_size > 0:
        root_transform = AffineTransform(compute_matrix_for_rescaling(width, height, target_size, align="center"))
        width, height = target_size, target_size

    normalized_paths_or_err: list[str | None] = [normalize_path_strings(x, root_transform) for x in raw_paths]
    normalized_paths: list[str] = [x for x in normalized_paths_or_err if x is not None]

    #split multipaths up
    paths = []
    for p in normalized_paths:
        p = ["M "+ x  for x in p.split('M') if x]
        paths.extend(p)

    return get_svg_string(paths, width, height)

def normalize_svg_topath(svg: str, target_size: int) -> list[Path]:
    width, height = get_svg_dimensions(svg)
    raw_paths = get_paths_with_transform(svg)



    root_transform= AffineTransform()
    if target_size > 0:
        root_transform = AffineTransform(compute_matrix_for_rescaling(width, height, target_size, align="center"))
        width, height = target_size, target_size

    normalized_paths_or_err: list[str | None] = [normalize_path_strings(x, root_transform) for x in raw_paths]
    normalized_paths: list[str] = [x for x in normalized_paths_or_err if x is not None]

    #split multipaths up
    paths = []
    for p in normalized_paths:
        p = [parse_path("M "+ x)  for x in p.split('M') if x]
        paths.extend(p)
    return paths


def normalize_svg_topath_rect(svg: str, target_w: float, target_h: float) -> list[Path]:
    width, height = get_svg_dimensions(svg)
    raw_paths = get_paths_with_transform(svg)

    root_transform = AffineTransform(compute_matrix_for_rescaling_rect(width, height, target_w, target_h, align="center"))

    normalized_paths_or_err: list[str | None] = [normalize_path_strings(x, root_transform) for x in raw_paths]
    normalized_paths: list[str] = [x for x in normalized_paths_or_err if x is not None]

    paths = []
    for p in normalized_paths:
        p = [parse_path("M " + x) for x in p.split('M') if x]
        paths.extend(p)
    return paths


def normalized_svg_frompaths(paths: list[Path], width: int, height: int, target_size: int):
    root_transform= AffineTransform()
    if target_size > 0:
        root_transform = AffineTransform(compute_matrix_for_rescaling(width, height, target_size, align="center"))
        width, height = target_size, target_size
    def func(points: np.ndarray):
        return root_transform.apply(points)
    normalized_paths: list[str] = [transform_path(p, func).d() for p in paths]

    return get_svg_string(normalized_paths, width, height)

def modify_thickness(svgstr, width: Callable):

    soup = BeautifulSoup(svgstr, "xml")

    for path in soup.find_all("path"):
        # Get existing style string
        style_str = path.get("style", "")
        style_dict = parse_style(style_str)

        # Modify stroke-width
        style_dict["stroke-width"] = str(width())
        style_dict["stroke-linecap"] = "round"

        # Convert back to string
        path["style"] = style_dict_to_string(style_dict)

    return str(soup)


def rescale_points(pts: np.ndarray, width, height, target_w, target_h):
    mat = AffineTransform(
        compute_matrix_for_rescaling_rect(width, height, target_w, target_h, align="center"))
    return mat.apply(pts)


if __name__ == "__main__":

    IN_PATH = os.path.join("tests",'5611.svg')
    OUT_PATH = os.path.join("tests",'normalized.svg')
    OUT_PATH2 = os.path.join("tests",'transforms_removed.svg')
    with open(IN_PATH, 'r') as f:
        svg = f.read()
    res = normalize_svg(svg, 800)
    res2 = normalize_svg(svg, -1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(res)
    with open(OUT_PATH2, "w", encoding="utf-8") as f:
        f.write(res2)
