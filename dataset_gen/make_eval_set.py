import argparse
import csv
import os
from glob import glob

import svgutils


TARGET_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Rasterize a folder of SVGs into target images for an eval set.")
    parser.add_argument("svg_dir", type=str, help="Folder containing input SVG files.")
    parser.add_argument("target_dir", type=str, help="Output folder for rasterized target images.")
    return parser.parse_args()


def make_target(svg_path: str, out_path: str):
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    svg = svgutils.normalize_svg(svg, target_size=TARGET_SIZE)
    modified = svgutils.modify_thickness(svg, width=lambda: 0.4)
    svgutils.save_webp_from_svg(modified, out_path, grayscale=True)


def main():
    args = parse_args()
    target_dir = os.path.normpath(args.target_dir)
    os.makedirs(target_dir, exist_ok=True)

    svg_paths = sorted(glob(os.path.join(args.svg_dir, "*.svg")))
    if not svg_paths:
        print(f"No SVG files found in {args.svg_dir}")
        return

    csv_dir = os.path.dirname(target_dir) or "."
    csv_name = os.path.basename(target_dir) + ".csv"
    csv_path = os.path.join(csv_dir, csv_name)

    rows = []
    for svg_path in svg_paths:
        stem = os.path.splitext(os.path.basename(svg_path))[0]
        out_path = os.path.join(target_dir, f"{stem}_output.webp")
        print("processing", svg_path)
        make_target(svg_path, out_path)
        rows.append((
            f"{stem}.png",
            f"{stem}_output.webp",
        ))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
