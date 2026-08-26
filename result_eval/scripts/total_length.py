import argparse

from svgutils import normalize_svg_topath_rect, getpoints_maxdist, get_svg_dimensions
from metrics import total_length


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print combined polyline length of an SVG (no resize).")
    p.add_argument("svg", help="Path to the SVG file")
    p.add_argument("--n", type=int, default=80,
                   help="Number of sample points per path segment for getpoints_maxdist()")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.svg, "r", encoding="utf-8") as f:
        svg_str = f.read()

    w, h = get_svg_dimensions(svg_str)
    paths = normalize_svg_topath_rect(svg_str, w, h)
    polylines = [getpoints_maxdist(p, args.n) for p in paths]

    print(total_length(polylines))


if __name__ == "__main__":
    main()
