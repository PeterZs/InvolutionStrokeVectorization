"""Convert hand-drawn SVG annotations (one <circle> per intersection point) into
the per-sample `x,y` CSV format used for the ground truth intersections, in bulk.

Each circle's centre (cx, cy) becomes one CSV row. Coordinates are taken as-is
from the SVG user space (the annotation SVGs use a viewBox whose longest edge is
1024, matching the annotation coordinate space), so no scaling is applied.

Deliberately minimal: only <circle> elements are read. transform attributes are
NOT supported -- any transform would move the points out of user space, so the
script raises instead of silently producing wrong coordinates.
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET


def _localname(tag: str) -> str:
    """Strip the XML namespace, e.g. '{http://...}circle' -> 'circle'."""
    return tag.rpartition("}")[2]


def svg_to_points(path: str) -> list[tuple[float, float]]:
    """Return [(cx, cy), ...] for every <circle> in the SVG, in document order.

    Raises if any element carries a transform attribute (unsupported).
    """
    root = ET.parse(path).getroot()
    for el in root.iter():
        if "transform" in el.attrib:
            raise ValueError(
                f"{path}: <{_localname(el.tag)}> has a transform attribute; "
                f"transforms are not supported."
            )
    points = []
    for el in root.iter():
        if _localname(el.tag) == "circle":
            points.append((float(el.get("cx")), float(el.get("cy"))))
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", help="Folder of annotation SVGs")
    ap.add_argument("output_dir", help="Folder to write the CSVs into "
                                       "(NOT the real annotation dir)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.svg")))
    if not paths:
        raise SystemExit(f"No SVG files found in {args.input_dir}")

    for path in paths:
        points = svg_to_points(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.csv")
        with open(out_path, "w", encoding="utf-8") as f:
            for x, y in points:
                f.write(f"{x:.2f},{y:.2f}\n")
        print(f"Wrote {out_path} ({len(points)} points)")


if __name__ == "__main__":
    main()
