import numpy as np
import cv2
import json
import os
import sys
import random
import itertools
from svgpathtools import svg2paths2, wsvg, Path, Line, svgstr2paths
import cairosvg
from typing import Any, Callable
# parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent)
import brushengine.brushengine as brushengine
import svgutils
import uuid
from PIL import Image
import utils
from dataclasses import asdict
import copy

def sample_description(id: str, b: brushengine.Brush | None = None, d = {}):
    
    if b is not None:
        d = asdict(b.params)
        d["name"] = b.name
    d['id'] = id
    return d

def save_img_webp_lossless(cv_img, output_path, grayscale=False):
    """
    Save a 4-channel OpenCV image to a lossless WebP file.

    Parameters:
        cv_img (numpy.ndarray): OpenCV image (expected 4 channels: BGRA)
        output_path (str): Path to save the WebP image
        grayscale (bool): If True, convert to grayscale before saving
    """
    # Drop the alpha channel
    bgr_img = cv_img[:, :, :3]

    # Convert BGR -> RGB for Pillow
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Create a PIL Image
    pil_img = Image.fromarray(rgb_img)

    # Convert to grayscale if requested
    if grayscale:
        pil_img = pil_img.convert('L')

    # Save as lossless WebP
    pil_img.save(output_path, format='WEBP', lossless=True)


def bounded_combinations(tmp: dict[str, Any], mxitems = 5):
    keys = list(tmp.keys())
    values = list(tmp.values())
    if mxitems >0:
        limited_combinations = itertools.islice(
            (dict(zip(keys, v)) for v in itertools.product(*values)),
            mxitems
        )
    else:
        limited_combinations = (dict(zip(keys, v)) for v in itertools.product(*values))
    return limited_combinations
    
def get_all_brushes(include: list[str] = []):
    brushfolder = 'brushengine/brushes'
    folders = [name for name in os.listdir(brushfolder)
           if os.path.isdir(os.path.join(brushfolder, name))]
    print(folders)
    print(include)
    return [brushengine.load_brush(os.path.join(brushfolder, bname)) for bname in folders
            if bname in include or not len(include)]

    
def _draw_single(points: list[np.ndarray],
         brush : brushengine.Brush,
         out_path : str,
         size = 1000,
         downsample_size=1000) -> bool:
    img = np.ones((int(size), int(size), 4), dtype=np.uint8) * 255
    
    print("    ", brush.params.size_pixels)
    for p in points:
        img= brushengine.draw_patch_along_polyline(img, p, brush.tips, brush.params, texture=brush.texture)
    # print("done")
    
    # Reject drawings with large solid foreground blobs: if a circle of this
    # diameter fits entirely inside a non-white (< 255) region, the drawing is
    # considered a failure. Erosion with a circular kernel checks exactly that.
    gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    foreground = (gray < 255).astype(np.uint8)
    diameter = max(50, 1.75 * brushengine.get_max_brush_size(brush))
    ksize = int(round(diameter))
    #print("kernel size:", ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(foreground, kernel)
    success = not bool(np.any(eroded))
    if not success:
        print("     invalid")
        # eroded pixels are valid centers for a fully-fitting circle; dilating
        # them back with the same kernel paints the union of those circles.
        circles = cv2.dilate(eroded, kernel)
        debug = img[..., :3].copy()
        debug[circles > 0] = (0, 0, 255)  # red overlay (BGR)
        debug_path = out_path[:-5] + "_debug.png"
        cv2.imwrite(debug_path, debug)
        return False

    target_size =(downsample_size, downsample_size)
    downsampled_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, downsampled_img)
    save_img_webp_lossless(downsampled_img, out_path)
    return True

#useful for debugging
def draw_single_separate(points: list[np.ndarray], 
         brush : brushengine.Brush, 
         out_path : str,
         size = 1000,
         downsample_size=1000):
    for idx, p in enumerate(points):
        img = np.ones((int(size), int(size), 4), dtype=np.uint8) * 255
        img= brushengine.draw_patch_along_polyline(img, p, brush.tips, brush.params, texture=brush.texture)
        target_size =(downsample_size, downsample_size)
        downsampled_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        #cv2.imwrite(out_path, downsampled_img)
        save_img_webp_lossless(downsampled_img, out_path[:-5]+f"{idx}.webp")
    
    

def generate_image_pairs(svgstr: str, 
         brushes: list[brushengine.Brush], 
         variation: dict,
         out_dir: str,
         naming: str,
         out_image_size: int ,
         num_variations : int= 5,
         append_log_file : str = "",
         needs_downscaling: list[str] = [],
         num_brushes_per_sample : int = -1,
         ):
    
    #step 1: generate id for this sample
    id = str(uuid.uuid4())
    
    #step 2: store the target centerline image
    outpath = os.path.join(out_dir, f"{naming}_output.webp")
    width = lambda  : 0.4
    modified = svgutils.modify_thickness(svgstr, width=width)
    svgutils.save_webp_from_svg(modified, outpath, grayscale=True)
    # with open(f"{out_dir}/debug.svg", "w") as f:
    #     f.write(modified)
    
    #step 3: randomly select which brushes to use
    all_brush_names = [b.name for b in brushes] + ['basic']
    active_brush_names = all_brush_names if num_brushes_per_sample <= 0 else random.sample(all_brush_names, num_brushes_per_sample)
    
    #step 4: load paths, convert them to dense polylines
    paths, attributes = svgstr2paths(svgstr)
    points = []
    points2x = []
    for p in paths:
        #sometimes there are empty paths
        if len(p):
            points.append(svgutils.getpoints(p, n=80))
            points2x.append(points[-1]*2)
    
    #step 5: generate input painted images
    counter = 0
    for b_original in brushes:
        #we need to copy the object because we modify brush attributes
        b = copy.deepcopy(b_original)
        if b.name not in active_brush_names:
            continue
        print("    processing", b.name)
        curr_points = points
        curr_image_size = out_image_size
        if b.name in needs_downscaling:
            curr_points = points2x
            curr_image_size = 2*out_image_size
        if not b.name in variation:
            name = f"{naming}_{counter}_{b.name}"
            counter +=1
            outpath = out_dir+f"/{name}.webp"
            #brush without variation
            success = _draw_single(curr_points, b, outpath, curr_image_size, out_image_size)
            if success and append_log_file: utils.append_dict(append_log_file, sample_description(name, b))
        else:
            # combinations = list(bounded_combinations(variation[b.name], mxitems=-1))
            # random.shuffle(combinations)
            # for i, comb in enumerate(combinations[:num_variations]):
            for k, v in variation[b.name].items():
                new_val = getattr(b.params, k) + np.random.uniform(v[0], v[1])
                setattr(b.params, k, new_val)
            name = f"{naming}_{counter}_{b.name}"
            counter+=1
            outpath = out_dir+f"/{name}.webp"
            #print(b.params.size_pixels)
            if b.params.size_pixels < 0:
                print(b.name)
                print(b.params)
                print(b)
                raise ValueError("brush size is <0")
            success = _draw_single(curr_points, b, outpath, curr_image_size, out_image_size)
            if success and append_log_file: utils.append_dict(append_log_file, sample_description(name, b))

    #special case: a small, basic brush of varying width. 
    # Much faster than with the brushengine. 
    if 'basic' in active_brush_names:
        print("    processing basic brush")
        width = lambda  : 2 + np.random.uniform(0, 9.0)
        modified = svgutils.modify_thickness(svgstr, width=width)
        name = f"{naming}_{counter}_{'basic'}"
        counter +=1
        outpath = out_dir+f"/{name}.webp"
        svgutils.save_webp_from_svg(modified, outpath, grayscale=True)
        if append_log_file: utils.append_dict(append_log_file, sample_description(name, d={"name": "basic_brush", "size_pixels": "random(2, 11)"}))
    

    
    
    