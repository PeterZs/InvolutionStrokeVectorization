from dotenv import load_dotenv
import numpy as np
import cv2
import matplotlib.colors as mcolors
import torch
import os
from contextlib import contextmanager
from time import perf_counter
from pathlib import Path
import sys
import json
import torch

CONFIG_PATH = "config.env"
REQUIRED_VARS=["CENTERLINE_MODEL", "INTERSECTION_MODEL"]

COLORS = ["tomato", "mediumblue", "grey", 
          "yellowgreen", "plum", "burlywood", "lightseagreen", "cornflowerblue", "darkgreen", "darkorchid"]
PAIRED_COLORS = [("brown", "tomato"), ("mediumblue", "dodgerblue"), ("darkorchid", "plum"),  
                 ("teal", "paleturquoise"), ("olive", "khaki")]

def color_to_bgr(color):
    """
    Convert a Matplotlib color to an OpenCV BGR tuple (uint8).

    Parameters
    ----------
    color : str or tuple
        Matplotlib color (e.g. 'red', '#ff0000', (0.5, 0.2, 0.1), etc.)

    Returns
    -------
    tuple
        (B, G, R) in range 0–255
    """
    # Convert to RGBA in 0–1 range
    rgba = mcolors.to_rgba(color)

    # Extract RGB, convert to 0–255, and reverse to BGR
    r, g, b = rgba[:3]
    return (int(b * 255), int(g * 255), int(r * 255))


def load_env():
    """Load environment variables from a file."""
    load_dotenv(CONFIG_PATH)
    print(os.environ)
    for i in REQUIRED_VARS:
        if i not in os.environ:
            raise ValueError(f"you need to have {REQUIRED_VARS} set, but missing: {i}")

@contextmanager
def timer(desc=""):
    result = {"time": -1.0}
    start = perf_counter()
    yield result
    result["time"] = perf_counter() - start
    if desc:
        print(f"{desc} time: {result['time']}")
        
def pil_to_cv2(pil_img):
    # Convert PIL image to numpy array
    cv_img = np.array(pil_img)
    
    # Handle different modes
    if pil_img.mode == "RGB":
        # PIL: RGB, OpenCV: BGR
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    elif pil_img.mode == "RGBA":
        # PIL: RGBA, OpenCV: BGRA
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
    elif pil_img.mode == "L":
        # Grayscale, no change needed
        pass
    else:
        raise ValueError("unsupported pil image mode:", pil_img.mode)
    
    return cv_img

def cv2_to_torch(img):
    """
    Convert OpenCV image (H, W, C, BGR, uint8) 
    -> Torch tensor (C, H, W, RGB, float32, 0-1)
    """
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32 and normalize to [0, 1]
    img = img.astype("float32") / 255.0

    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1)

    return img


def resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if input_path.is_dir():
        return input_path

    return input_path.parent


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        # Extend extensions as needed
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        return [p for p in input_path.iterdir() if p.suffix.lower() in exts]

    raise ValueError(f"Invalid input path: {input_path}")


def output_name(image_path: Path, output_dir: Path, suffix = "output") -> Path:
    """
    Generate output filename based on input name.

    Example:
        input:  photo.jpg
        output: photo_output.png
    """
    return output_dir / f"{image_path.stem}_{suffix}"

def save_polylines_to_json(lines: dict[int, np.ndarray], path: str) -> None:
    serialized_lines = {}

    for k, val in lines.items():

        # Convert numpy array → python list while preserving order
        serialized_lines[k] = val.tolist()

    # Write JSON (lists preserve order by definition)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized_lines, f, ensure_ascii=False)

def get_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")