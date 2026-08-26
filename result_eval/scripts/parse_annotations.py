"""Parse a CVAT annotations.xml and write one CSV per image with the raw point coordinates."""

import argparse
import os
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-image intersection points from annotations.xml.")
    parser.add_argument("annotations", help="Path to annotations.xml")
    parser.add_argument("output_dir", help="Directory to write the per-image CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tree = ET.parse(args.annotations)
    root = tree.getroot()

    for image in root.findall("image"):
        name = image.get("name", "")
        stem = os.path.splitext(os.path.basename(name))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.csv")

        with open(out_path, "w", encoding="utf-8") as f:
            for pt in image.findall("points"):
                pts_attr = pt.get("points", "")
                # CVAT points attr is "x,y" (single) or "x1,y1;x2,y2;..."
                for pair in pts_attr.split(";"):
                    pair = pair.strip()
                    if pair:
                        f.write(pair + "\n")

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
