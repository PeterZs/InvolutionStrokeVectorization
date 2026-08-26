import xml.etree.ElementTree as ET
import numpy as np
import cairosvg
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

def parse_style(style_str):
    """Convert a style string into a dictionary"""
    style_dict = {}
    if style_str:
        for item in style_str.split(";"):
            if item.strip():
                key, value = item.split(":")
                style_dict[key.strip()] = value.strip()
    return style_dict

def style_dict_to_string(style_dict):
    """Convert a dictionary back to a style string"""
    return "; ".join(f"{k}: {v}" for k, v in style_dict.items())


# Load SVG
tree = ET.parse("testcase/bigtest.svg")


root = tree.getroot()

# Modify stroke width of all paths
for path in root.findall(".//svg:path", SVG_NS):
    style = parse_style(path.get("style", ""))
    style["stroke-width"] = "10"
    path.set("style", style_dict_to_string(style))

# Save modified SVG
tree.write("testcase/modified.svg")
cairosvg.svg2png(url="testcase/modified.svg", 
                 write_to="testcase/output.png", 
                 output_height=1000, 
                 output_width=1000, 
                 background_color="white")
cairosvg.svg2png(url="testcase/bigtest.svg", 
                 write_to="testcase/thin.png", 
                 output_height=2000, 
                 output_width=2000, 
                 background_color="white")
