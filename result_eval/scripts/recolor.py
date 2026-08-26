"""Recolor every stroke in an SVG by cycling through a fixed palette.

Naive string-level rewrite: find each `stroke="..."` attribute and replace
its value with the next color from COLORS.
"""

import argparse
import os
import re

COLORS = ["tomato", "mediumblue", "grey",
          "yellowgreen", "plum", "burlywood", "lightseagreen", "cornflowerblue", "darkgreen", "darkorchid"]

STROKE_RE = re.compile(r'stroke="[^"]*"')


def recolor(svg_text: str) -> str:
    counter = {"i": 0}

    def replace(_match):
        color = COLORS[counter["i"] % len(COLORS)]
        counter["i"] += 1
        return f'stroke="{color}"'

    return STROKE_RE.sub(replace, svg_text)


def main():
    parser = argparse.ArgumentParser(description="Recolor SVG strokes by cycling through a palette.")
    parser.add_argument("input", help="Path to input SVG file")
    parser.add_argument("-o", "--output", help="Output path (default: <input_stem>_recolored.svg in CWD)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        svg = f.read()

    recolored = recolor(svg)

    if args.output:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(os.getcwd(), f"{stem}_recolored.svg")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(recolored)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
