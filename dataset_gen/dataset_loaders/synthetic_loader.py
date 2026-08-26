import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils

from curve_intersections.svgcreator import (IntersectingCurveGenerator,
        create_svg_path_from_bezier, single_intersects, complex_to_array)
import svgutils

import random
import numpy as np
import math
from typing import Literal

from dataset_loaders.loader_utils import Loader



MAX_TRIES = 50

def bezier_to_path(ctrl_pts):
    pts = complex_to_array(ctrl_pts)
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    x3, y3 = pts[3]
    return f"M {x0:.6g} {y0:.6g} C {x1:.6g} {y1:.6g} {x2:.6g} {y2:.6g} {x3:.6g} {y3:.6g}"

def get_partition(start, stop, n, min_len=0):
    """
    Partitions the float interval [start, stop] into n random intervals.
    
    Args:
        start (float): The starting value of the interval.
        stop (float): The ending value of the interval.
        n (int): The number of desired intervals.
        min_len (float): The minimum length of each interval.

    Returns:
        list: A list of n floats representing the start of each interval. 
              The first value is always 'start'.
    
    Raises:
        ValueError: If the total interval length is insufficient for n * min_len.
    """
    if n < 1:
        raise ValueError("Number of intervals (n) must be at least 1.")
        
    width = stop - start
    required_width = n * min_len
    
    if required_width > width:
        raise ValueError(f"Total interval width ({width}) is too small for "
                         f"{n} intervals of minimum length {min_len}.")

    # Calculate the "slack" (the length available to be randomized)
    slack = width - required_width

    # Generate n-1 random points within the slack and sort them.
    # We add 0.0 at the start to represent the beginning of the slack allocation.
    slack_cuts = [0.0] + sorted(random.uniform(0, slack) for _ in range(n - 1))

    # Calculate the actual coordinates.
    # Formula: start + (random_slack_point) + (index * fixed_min_len_bias)
    partition_points = [
        start + cut + (i * min_len) 
        for i, cut in enumerate(slack_cuts)
    ]

    return partition_points

def _get_line_star(n, center, radius, min_angle):
    """
    Returns n random lines joining in the center.
    
    Args:
        n (int): Number of lines (rays).
        center (tuple): The (x, y) coordinates of the star's center.
        radius (float): The length of the lines.
        min_angle (float): The minimum angle between adjacent lines (in radians).

    Returns:
        list[str]: A list of SVG 'd' path strings.
    """
    # 1. Determine the angles.
    # We treat the circle as an interval from 0 to 2*PI.
    # get_partition guarantees the first angle is 0 and respecting min_angle.
    # The constraint ensures the last angle also leaves enough room before wrapping back to 0.
    min_angle = np.deg2rad(min_angle)
    angles = get_partition(0, 2 * math.pi, n, min_angle)

    cx, cy = center
    paths = []

    for angle in angles:
        # 2. Calculate endpoint using polar to cartesian conversion
        end_x = cx + radius * math.cos(angle)
        end_y = cy + radius * math.sin(angle)
        
        # 3. Format as SVG Path d-string (Move to center, Line to endpoint)
        d_string = f"M {cx} {cy} L {end_x} {end_y}"
        paths.append(d_string)
        
    return paths


def get_line_star(n, radius,  min_angle, svgsize=1000):
    
    center = svgsize//2
    paths = _get_line_star(n, (center, center), radius, min_angle)
    
    angle = random.uniform(0, 359)
    transform = svgutils.AffineTransform.from_svg_matrix(f"rotate({angle}, {center}, {center})")
    paths = [svgutils.normalize_path_strings(([], x), transform) for x in paths]
    return svgutils.get_svg_string(paths, svgsize, svgsize)
    
    

def _get_curvy_star(angles, svgsize=1000):
    assert angles[0] == 0
    assert angles[-1] < 180 
    
    curves = []
    g = IntersectingCurveGenerator(svg_size=svgsize)
    l = 800
    
    while True:
        g.create_main_curve(l)
        if not g.curve1.cx:
            break
        print("self-inter (main)")
    
    curves.append(g.curve1)
    print("main curve created")

    idx = 1

    while idx < len(angles):
        angle = angles[idx]
        success = False

        # Try up to max_tries to find a valid curve for this angle
        for _ in range(MAX_TRIES):
            g.create_second_curve(l, angle)
            
            # Check if the new candidate intersects ALL existing curves exactly once
            if all(single_intersects(c, g.curve2) for c in curves):
                curves.append(g.curve2)
                success = True
                break
        
        if success:
            idx += 1
        else:
            print(f"Failed to match angle {idx} after {MAX_TRIES} tries.")
            
            print("Backtracking to regenerate previous curve...")
            idx -= 1
            curves.pop() 
    return curves
      

def get_curvy_star(n, min_angle, strategy: Literal["symmetric", "evenodd", "meet_on_line", "sink"] = "evenodd", svgsize=1000):
    
    assert n <= 9, "9 stripes are the maximum supported"
    print("generating curvy star with", n, "stripes and strategy:", strategy)
    odd = n%2==1
    if strategy in ["evenodd", "sink"]:
        tmp = n
    elif strategy == "meet_on_line":
        tmp = n-1
    else:
        tmp = n//2 + odd
    anglerange, min_angle = (180, min_angle) if strategy != "sink" else (90, min_angle/2)

    angles =  get_partition(0, anglerange, tmp, min_angle)
    curves = _get_curvy_star(angles, svgsize)
    if strategy == "evenodd":
        #alternatingly keep start half and end half
        curves = [c[(i%2)*0.5: (i%2)*0.5+0.5] for i, c in enumerate(curves)]
    elif strategy == "sink":
        curves = [c[0:0.5] for i, c in enumerate(curves)]
    elif strategy == "meet_on_line":
        #keep the first curve symmetric, keep start half for the rest
        curves = [curves[0]] + [c[0:0.5] for i, c in enumerate(curves[1:])]
    else:
        if odd:
            curves[-1] = curves[-1][0:0.5]
            
    paths = [bezier_to_path(c.v) for c in curves]
    angle = random.uniform(0, 359)
    center = svgsize//2
    transform = svgutils.AffineTransform.from_svg_matrix(f"rotate({angle}, {center}, {center})")
    paths = [svgutils.normalize_path_strings(([], x), transform) for x in paths]
    return svgutils.get_svg_string(paths, svgsize, svgsize)


class SyntheticDataset(Loader):
    
    
    def sample_loader(self, split: str, target_size, index = 0):
        MIN_ANGLE = 15
        
        random.seed(index)
        
        while True:
            num_stripes = random.randint(4, 9)
            #30% of line stars, 70% of curvy stars
            if index in [0, 3, 7]:
                num_stripes = random.randint(4, 9)
                radius = random.randint(400, 1005)
                yield get_line_star(num_stripes, radius, MIN_ANGLE)
            else:
                strategy = random.choice(["symmetric", "evenodd", "meet_on_line", "sink"])
                if strategy == "sink":
                    num_stripes = random.randint(3, 6)
                else:
                    num_stripes = random.randint(4, 9)
                yield get_curvy_star(num_stripes, MIN_ANGLE, strategy)
            index+=1
        
        