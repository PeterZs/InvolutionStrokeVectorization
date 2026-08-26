import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from imgpreprocess import preprocess


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def pad_to_square(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    h, w = arr.shape[:2]
    side = max(h, w)
    new_h = side
    new_w = side
    if (new_h, new_w) == (h, w):
        return image
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if arr.ndim == 3:
        pad_width.append((0, 0))
    padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return Image.fromarray(padded, mode=image.mode)


def process_file(in_path: Path, out_path: Path, invert: bool, skip_norm: bool) -> None:
    image = Image.open(in_path).convert("L")
    result = preprocess(image, invert=invert, skip_norm=skip_norm)
    result = pad_to_square(result)
    if invert:
        result = ImageOps.invert(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess all images in a folder and pad to a square (max of width/height)."
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing input images")
    parser.add_argument("output_folder", type=Path, help="Folder to write processed images")
    parser.add_argument("--no-invert", action="store_true", help="Disable inversion in preprocess")
    parser.add_argument("--skip-norm", action="store_true", help="Skip normalization in preprocess")
    args = parser.parse_args()

    in_dir: Path = args.input_folder
    out_dir: Path = args.output_folder

    if not in_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {in_dir}")

    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {in_dir}")
        return

    for f in files:
        out_path = out_dir / f.name
        try:
            process_file(f, out_path, invert=not args.no_invert, skip_norm=args.skip_norm)
            print(f"  {f.name} -> {out_path}")
        except Exception as e:
            print(f"  failed {f.name}: {e}")


if __name__ == "__main__":
    main()
