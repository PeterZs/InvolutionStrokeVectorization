import random
from abc import ABC, abstractmethod

import numpy as np
import svgwrite

from .bezier import bezier


def complex_to_array(c):
    return np.array([c.real, c.imag]).T


def array_to_complex(a):
    return a[0] + a[1] * 1j


def bezier_from_points(p1, p2, p3, p4):
    p1 = array_to_complex(p1)
    p2 = array_to_complex(p2)
    p3 = array_to_complex(p3)
    p4 = array_to_complex(p4)
    return bezier(p1, p2, p3, p4)


def create_svg_path_from_bezier(curve, color="black", stroke_width=10):
    # Convert the control points to an array
    cp = complex_to_array(curve.v)

    # Create the path
    path = svgwrite.path.Path()
    path.push("M", cp[0])  # Move to the beginning of the curve
    path.push("C", *cp[1:])  # Draw the curve

    # Set style attributes
    path.update({"stroke": color, "stroke-width": stroke_width, "fill": "none"})
    return path

def single_intersects(curve1: bezier, curve2: bezier, val=0.5, tol = 5e-3):
        # If one of the curves self interesects then return True
        if len(curve1.cx) > 0 or len(curve2.cx) > 0:
            return False

        # Compute the number of intersections between the two curves
        # Find the ones that are not at the middle of the curve
        all_intersection_points = curve1.x_bez(curve2)  # All intersection points
        all_intersection_points = all_intersection_points - np.array([val, val])
        number_other_intersections = (
            np.linalg.norm(all_intersection_points, axis=1, ord=np.inf) > tol
        ).sum()  # Number of intersections that are not at the middle of the curve

        # Return True if there is no other intersections
        if number_other_intersections > 0:
            return False

        return True


class CurveGenerator(ABC):
    """Base class to generate a set to curves that either intersect or are tangent to each other."""

    def __init__(self, svg_size=500) -> None:
        self.svg_size = svg_size
        self.center = np.array([self.svg_size // 2, self.svg_size // 2])
        self.distance = np.sqrt(2) * self.svg_size / 2  # Distance of inscribed circle

    @abstractmethod
    def get(self):
        pass

    def create_main_curve(self, tangent_length):
        ### Create first control point p1
        self.angle1 = random.uniform(90, 270)  # p1 should be on the left side
        angle_rad1 = np.deg2rad(self.angle1)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance1 = self.distance * random.uniform(1.5, 2)

        distance1 = self.distance * 1.5

        self.p1 = self.center + distance1 * np.array([np.cos(angle_rad1), np.sin(angle_rad1)])

        ### Create the last control point p4
        self.angle2 = random.uniform(270, 450)  # p4 should be on the right side
        angle_rad2 = np.deg2rad(self.angle2)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance4 = self.distance * random.uniform(1.5, 2)

        distance4 = self.distance * 1.5

        self.p4 = self.center + distance4 * np.array([np.cos(angle_rad2), np.sin(angle_rad2)])

        self.p2, self.p3 = self.from_14(
            self.p1, self.p4, tangent_angle_degree=0, tangent_length=tangent_length
        )

        self.curve1 = bezier_from_points(self.p1, self.p2, self.p3, self.p4)

    @abstractmethod
    def create_second_curve(self):
        pass

    def single_intersects(self):
        # If one of the curves self interesects then return True
        if len(self.curve1.cx) > 0 or len(self.curve2.cx) > 0:
            return False

        # Compute the number of intersections between the two curves
        # Find the ones that are not at the middle of the curve
        all_intersection_points = self.curve1.x_bez(self.curve2)  # All intersection points
        all_intersection_points = all_intersection_points - np.array([0.5, 0.5])
        number_other_intersections = (
            np.linalg.norm(all_intersection_points, axis=1, ord=np.inf) > 5e-3
        ).sum()  # Number of intersections that are not at the middle of the curve

        # Return True if there is no other intersections
        if number_other_intersections > 0:
            return False

        return True

    def from_14(self, p1, p4, tangent_angle_degree=0, tangent_length=100):
        """
        Create control points 2 and 3 from control points 1 and 4,
        position of the center of the curve and tangent value.
        """

        tangent_angle = np.deg2rad(tangent_angle_degree)
        tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

        p2 = (8 * self.center - 4 * tangent_length * tangent - 4 * p1 + 2 * p4) / 6
        p3 = (8 * self.center + 4 * tangent_length * tangent + 2 * p1 - 4 * p4) / 6

        return p2, p3

    def save_svg(self, filename, color1="black", color2="black", stroke_width=10):
        dwg = svgwrite.Drawing(filename, size=(self.svg_size, self.svg_size))

        path1 = create_svg_path_from_bezier(self.curve1, color1, stroke_width)
        path2 = create_svg_path_from_bezier(self.curve2, color2, stroke_width)

        dwg.add(path1)
        dwg.add(path2)

        # # Draw the center where the curves should meet
        # dwg.add(svgwrite.shapes.Circle(center=((250, 250)), r=8, fill="yellow"))

        dwg.save()

    def get_control_points(self):
        return dict(
            curve1=dict(p1=list(self.p1), p2=list(self.p2), p3=list(self.p3), p4=list(self.p4)),
            curve2=dict(p1=list(self.ip1), p2=list(self.ip2), p3=list(self.ip3), p4=list(self.ip4)),
        )


class IntersectingCurveGenerator(CurveGenerator):
    def __init__(self, svg_size=500) -> None:
        super().__init__(svg_size)

    def get(self, tangent_l= None, angle= None):
        
        self.tangent_length = random.randint(500, 800) if tangent_l is None else tangent_l
        self.intersecting_angle = random.randint(1, 90) if angle is None else angle

        self.create_main_curve(self.tangent_length)
        self.create_second_curve(self.tangent_length, self.intersecting_angle)

        while not self.single_intersects():
            self.create_main_curve(self.tangent_length)
            self.create_second_curve(self.tangent_length, self.intersecting_angle)
            
    
    # def get_multiple(self, angles : list[int]):
        
        # self.tangent_length = random.randint(500, 800)
        # self.create_main_curve(self.tangent_length)
        
        
        # self.create_second_curve(self.tangent_length, self.intersecting_angle)
        
        
        

    def create_second_curve(self, tangent_length=500, intersecting_angle=1):
        """Create the second curve that intersects the first curve.

        Parameters
        ----------
        intersecting_angle : int, optional
            Angle in degree, between 1 and 90, by default 1
        tangent_length : int, optional
            Tangent length, by default 500
        """

        ### Create first control point p1
        self.angle_ip1 = random.uniform(self.angle1, self.angle2)  # p1 should be on the top
        angle_rad_ip1 = np.deg2rad(self.angle_ip1)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance1 = self.distance * random.uniform(1.5, 2)

        distance1 = self.distance * 1.5

        self.ip1 = self.center + distance1 * np.array(
            [np.cos(angle_rad_ip1), np.sin(angle_rad_ip1)]
        )

        ### Create the last control point p4
        self.angle_ip4 = random.uniform(
            self.angle2, self.angle1 + 360
        )  # p4 should be on the bottom
        angle_rad_ip4 = np.deg2rad(self.angle_ip4)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance4 = self.distance * random.uniform(1.5, 2)

        distance4 = self.distance * 1.5

        self.ip4 = self.center + distance4 * np.array(
            [np.cos(angle_rad_ip4), np.sin(angle_rad_ip4)]
        )

        self.ip2, self.ip3 = self.from_14(
            self.ip1,
            self.ip4,
            tangent_angle_degree=intersecting_angle,
            tangent_length=tangent_length,
        )

        self.curve2 = bezier_from_points(self.ip1, self.ip2, self.ip3, self.ip4)


class TangentialCurveGenerator(CurveGenerator):
    def __init__(self, svg_size=500) -> None:
        super().__init__(svg_size)

    def get(self):
        self.tangent_length = random.randint(500, 800)

        self.create_main_curve(self.tangent_length)
        self.create_second_curve(self.tangent_length)

        while not self.single_intersects():
            self.create_main_curve(self.tangent_length)
            self.create_second_curve(self.tangent_length)
        

    def create_second_curve(self, tangent_length=500):
        """Create the second curve that intersects the first curve.

        Parameters
        ----------
        tangent_length : int, optional
            Tangent length, by default 500
        """

        ### Create first control point p1
        self.angle_ip1 = random.uniform(self.angle1, self.angle2)  # p1 should be on the top
        angle_rad_ip1 = np.deg2rad(self.angle_ip1)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance1 = self.distance * random.uniform(1.5, 2)
        self.ip1 = self.center + distance1 * np.array(
            [np.cos(angle_rad_ip1), np.sin(angle_rad_ip1)]
        )

        ### Create the last control point p4
        self.angle_ip2 = random.uniform(self.angle_ip1, self.angle2)  # p4 should be on the top
        angle_rad_ip2 = np.deg2rad(self.angle_ip2)

        # Add some randomness to the distance from the center,
        # Also make sure that we can rotate and translate the curve as much as we want and the curve
        # will remain the fully in the image
        distance2 = self.distance * random.uniform(1.5, 2)
        self.ip4 = self.center + distance2 * np.array(
            [np.cos(angle_rad_ip2), np.sin(angle_rad_ip2)]
        )

        self.ip2, self.ip3 = self.from_14(
            self.ip1, self.ip4, tangent_angle_degree=0, tangent_length=tangent_length
        )

        self.curve2 = bezier_from_points(self.ip1, self.ip2, self.ip3, self.ip4)
