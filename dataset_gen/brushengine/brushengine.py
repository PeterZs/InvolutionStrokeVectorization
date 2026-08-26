import cv2
import numpy as np
import math
import json
import os
from typing import NamedTuple, Any
from dataclasses import dataclass
from scipy.stats import truncnorm
import pathlib
import random

def bounded_normal_stream(mu=0.0, sigma=1.0, low=-1.0, high=2.0, chunk_size=1000):
    """Generator that yields samples from a truncated normal on demand."""
    a, b = (low - mu) / sigma, (high - mu) / sigma
    while True:
        yield from truncnorm.rvs(a, b, loc=mu, scale=sigma, size=chunk_size)

@dataclass
class BrushPhase:
    fraction: float
    size_profile: list[float] | None = None
    opacity_profile: list[float] | None = None

@dataclass
class WholeBrushPhase:
    opacity_profile: list[float]

@dataclass
class BrushParams:
    size_pixels:int
    distance_fraction: float
    size_spread_fraction: float
    x_spread: float
    y_spread: float
    base_rotation: float
    rotation_spread_fraction: float
    base_patch_opacity: float
    patch_opacity_spread: float
    texture_mode: int
    start_phase : BrushPhase | None
    end_phase: BrushPhase | None
    opacity: float = 1.0
    opacity_phase: WholeBrushPhase | None = None
    
class Brush(NamedTuple):
    name: str
    params: BrushParams
    tips: list[np.ndarray]
    texture: np.ndarray | None


def get_max_brush_size(brush: "Brush") -> float:
    """Largest footprint a single brush patch can occupy.

    A patch is scaled by `curr_size = size_factor * size_pixels`, where
    `size_factor` is either a phase `size_profile` value or, outside the
    phases, `1 + size_spread_fraction` (see `draw_patch_along_polyline`).
    On top of that, the positional jitter widens the stroke band by
    `x_spread`/`y_spread`: `half_band = 0.5 * spread * curr_size` on each
    side, so the total covered extent is `curr_size * (1 + spread)`.
    """
    params = brush.params
    # maximum size factor: phase profiles, or the spread-driven default
    size_factors = [1.0 + params.size_spread_fraction]
    for phase in (params.start_phase, params.end_phase):
        if phase is not None and phase.size_profile is not None:
            size_factors.extend(phase.size_profile)
    max_size = params.size_pixels * max(size_factors)
    # positional jitter widens the footprint along the larger spread axis
    return max_size * (1.0 + max(params.x_spread, params.y_spread))


def load_brush(brush_dir: str):
    p = pathlib.Path(brush_dir)
    with open(str(p / "config.json"), 'r') as f:
        data = json.load(f)
    
    if data["start_phase"] is not None:
        data["start_phase"] = BrushPhase(**data["start_phase"])
    if data["end_phase"] is not None:
        data["end_phase"] = BrushPhase(**data["end_phase"])
    if data.get("opacity_phase") is not None:
        data["opacity_phase"] = WholeBrushPhase(data["opacity_phase"])
    brush_params = BrushParams(**data)
    
    tips = []
    for name in os.listdir(brush_dir):
        if name.startswith("tip") and name.endswith(".png"):
            tips.append(
                cv2.imread(os.path.join(brush_dir,name), cv2.IMREAD_UNCHANGED)[..., 3]
            )
    if not tips:
        size = 64
        img = np.zeros((math.ceil(size +6 ), math.ceil(size +6 )), dtype=np.uint8)
        center = int(size/2 +2 ), int(size/2 +2)
        tip = cv2.circle(img, center, radius= int(size/2), 
                         color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        coords = np.argwhere(tip)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Slice that extracts the bounding box
        tip= tip[y_min:y_max+1, x_min:x_max+1]
        tips.append(tip)
    
    texture = None
    if os.path.exists(os.path.join(brush_dir,'texture.png')):
        texture = cv2.imread(os.path.join(brush_dir,'texture.png'), cv2.IMREAD_UNCHANGED)[..., 3]
    return Brush(p.name, brush_params, tips, texture)


def linear_interpolate(arr: list, u: float) -> float:
    """
    Linearly interpolates within a 1D NumPy array based on u ∈ [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        1D input array.
    u : float
        Position in [0, 1] representing the relative location along the array.

    Returns
    -------
    float
        Interpolated value.
    """
    n = len(arr)
    # Convert u to a fractional index in [0, n-1]
    idx = u * (n - 1)
    i = math.floor(idx)
    f = idx - i 

    # Handle edge case when u == 1
    if i >= n - 1:
        return float(arr[-1])
    
    return float(arr[i] * (1 - f) + arr[i + 1] * f)

# def handle_start_end_size(currdist, totaldist, a, b):
#     if currdist < a:
#         u = 
        
    

def polyline_tangent_angle(points):
  """
  Calculates the local tangent angle for each segment of a polyline.

  The angle is calculated for the segment starting from points[i] to points[i+1].
  A horizontal line segment pointing to the right corresponds to an angle of 0.
  The angles are returned in radians, in the range [-pi, pi].

  Args:
    points: A numpy array of shape (n, 2), where n is the number of points
            in the polyline.

  Returns:
    A numpy array of shape (n-1,) containing the tangent angle for each
    segment of the polyline in radians.
  """
  # Calculate the direction vectors for each segment
  direction_vectors = np.diff(points, axis=0)
  
  # Extract the x and y components of the direction vectors
  dx = direction_vectors[:, 0]
  dy = direction_vectors[:, 1]
  
  # Calculate the tangent angle using arctan2 for each segment
  angles = np.arctan2(dy, dx)
  
  return angles

def tile_image(image: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """
    Tile a NumPy image across a larger area.

    Parameters:
        image (np.ndarray): Input image of shape (h, w, c) or (h, w).
        out_height (int): Desired output height.
        out_width (int): Desired output width.

    Returns:
        np.ndarray: Tiled image of shape (out_height, out_width, c) or (out_height, out_width).
    """
    h, w = image.shape[:2]
    c = image.shape[2:]  # may be () or (channels,)

    # Compute number of repeats needed
    reps_y = -(-out_height // h)  # ceiling division
    reps_x = -(-out_width // w)

    # Tile image
    tiled = np.tile(image, (reps_y, reps_x, *([1] if c else [])))

    # Crop to exact output size
    tiled = tiled[:out_height, :out_width, ...]
    return tiled


def rotate_img(img, angle_deg):
    
    theta = math.radians(angle_deg)
    h, w = img.shape[:2]
    new_w =  w * abs(math.cos(theta)) + h * abs(math.sin(theta))
       
    new_h = w * abs(math.sin(theta)) + h * abs(math.cos(theta))

    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1)
    rotation_matrix[0, 2] += -w/2 + new_w/2 + 3
    rotation_matrix[1, 2] += -h/2 + new_h/2 + 3
    rotated = cv2.warpAffine(img, rotation_matrix, (math.ceil(new_h)+5, math.ceil(new_w)+5))
    # cv2.imwrite("rotated_.png", rotated)
    # print(rotated.shape)
    single_channel = len(rotated.shape) ==2 
    # print(single_channel)
    coords = np.argwhere(rotated if single_channel else rotated[..., -1])
    
    #happens when image is completely transparent / completely black
    if not len(coords):
        y_min, x_min = 0, 0
        y_max, x_max = math.ceil(new_h), math.ceil(new_w)
    else: 
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

    # Slice that extracts the bounding box
    rotated = rotated[y_min:y_max+1, x_min:x_max+1]

    return rotated


def resize_subpixel(patch: np.ndarray, scale:float):
    M = np.array([
        [scale, 0, 1],
        [0, scale, 1]
    ], dtype=np.float32)

    h, w = patch.shape[:2]
    # Warp only the visible region (antialiased subpixel placement)
    region_w = int(h*scale) + 3
    region_h = int(w*scale) + 3
    warped = cv2.warpAffine(
        patch, M, (region_w, region_h),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped


def patch_transform(patch: np.ndarray, canvas_w : int, canvas_h: int, 
                    center_xy: tuple[float, float], angle_deg: float, scale: float):
    
    h, w = patch.shape[:2]
    
    theta = math.radians(angle_deg)
    h, w = patch.shape[:2]
    rot_w =  w * abs(math.cos(theta)) + h * abs(math.sin(theta))
       
    rot_h = w * abs(math.sin(theta)) + h * abs(math.cos(theta))

    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1)
    #add offset for antialiasing
    rotation_matrix[0, 2] +=  -w/2 + rot_w/2
    rotation_matrix[1, 2] +=  -h/2 + rot_h/2
    M_rotate = np.vstack([rotation_matrix, np.array([0, 0, 1])])
    
    cx, cy = center_xy

    # Compute top-left corner of patch in canvas coordinates
    top_left_x = cx - (scale * rot_w) / 2 - 1 
    top_left_y = cy - (scale * rot_h) / 2 - 1

    # Determine bounding box intersection with canvas
    x0 = int(np.floor(top_left_x))
    y0 = int(np.floor(top_left_y))
    
    # Compute affine transform for subpixel positioning
    M_translate = np.array([
        [1, 0, top_left_x - x0],
        [0, 1, top_left_y - y0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    M_scale = np.array([
        [scale, 0, 1],
        [0, scale, 1],
        [0, 0, 1]
    ], dtype=np.float32)
    
    M_total = (M_scale @ M_translate @ M_rotate)[:2, :]
    
    new_w = math.ceil(rot_w * scale) + 5
    new_h = math.ceil(rot_h * scale) + 5
    
    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(canvas_w, x0 + new_w)
    clip_y1 = min(canvas_h, y0 + new_h)
    
    # print(top_left_x, top_left_y)
    # print(x0, y0)
    # print(rot_w, rot_h)
    # print(new_w, new_h)
    # print(clip_x0, clip_x1)
    # print(clip_y0, clip_y1)
    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
        return np.array([]), slice(0, 0), slice(0, 0)
    
    warped = cv2.warpAffine(
        patch, M_total, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # Crop to the visible area
    clipped_img = warped[
        (clip_y0 - y0):(clip_y1 - y0),
        (clip_x0 - x0):(clip_x1 - x0)
    ]
    # Corresponding canvas slice
    y_slice = slice(clip_y0, clip_y1)
    x_slice = slice(clip_x0, clip_x1)

    return clipped_img, y_slice, x_slice
    
    

def get_current_patch(patch: np.ndarray, new_size: float, angle_degrees: float) -> np.ndarray:
    """
    Applies scaling and rotation to a patch.
    """
    upscaled_size = max(1.0, new_size)
    scale = upscaled_size/float(patch.shape[0])

    # subpixel resizing
    # patch = cv2.resize(patch, (int(np.ceil(upscaled_size)), int(np.ceil(upscaled_size))), 
    #                     interpolation=cv2.INTER_AREA)
    
    patch = resize_subpixel(patch, scale)
    # print(patch.shape)
    rotated = rotate_img(patch, angle_degrees)
    # cv2.imwrite("transformed.png", 255-rotated)

    return rotated


def add_alpha_to_color(img_color, alpha):
    assert len(img_color.shape) ==3 and img_color.shape[-1] == 3
    assert len(alpha.shape) == 2
    assert img_color.shape[:2] == alpha.shape
    return np.concatenate([img_color, alpha[..., np.newaxis]], axis=2)


def alpha_blend_4channel(fg, bg):
    """
    Alpha composite fg image over bg image.
    Both images must be RGBA and the same size.
    Returns an RGBA image.
    """
    # Convert to float in [0,1]
    fg = fg.astype(float) / 255.0
    bg = bg.astype(float) / 255.0

    # Extract alpha channels
    alpha_f = fg[..., 3]
    alpha_b = bg[..., 3]

    # Compute output alpha
    alpha_out = alpha_f + alpha_b * (1 - alpha_f)

    # Compute output color channels
    out_rgb = np.zeros_like(fg[..., :3])
    for c in range(3):
        out_rgb[..., c] = (
            fg[..., c] * alpha_f + bg[..., c] * alpha_b * (1 - alpha_f)
        ) / (alpha_out + 1e-8)  # avoid div by zero

    # Combine RGB + alpha
    out = np.dstack((out_rgb, alpha_out))

    # Back to uint8
    return np.round(out * 255).clip(0, 255).astype(np.uint8)

def blend_masks(alpha_f, alpha_b):
    """
    Blend only the alpha channel of two RGBA images using the 'over' rule.
    Returns a copy of bg with updated alpha channel.
    """
    out = np.zeros_like(alpha_b)
    # Make a copy of the background to not modify original
    alpha_f = alpha_f.astype(float) / 255.0
    alpha_b = alpha_b.astype(float) / 255.0
    
    # Apply the 'over' formula to alpha only
    out = alpha_f * alpha_b
    
    # Convert back to uint8
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out

def add_masks(alpha_f, alpha_b):
    
    out = np.zeros_like(alpha_b)
    # Make a copy of the background to not modify original
    alpha_f = alpha_f.astype(float) / 255.0
    alpha_b = alpha_b.astype(float) / 255.0
    
    # Apply the 'over' formula to alpha only
    out = alpha_f + alpha_b * (1 - alpha_f)
    
    # Convert back to uint8
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out


def get_subpixel_patch_pos(patch, center_xy, canvas_w, canvas_h):
    """
    Places a patch onto a (conceptual) larger canvas with subpixel precision,
    but only renders and returns the minimal bounding region.

    Parameters
    ----------
    patch : np.ndarray
        The small image patch, shape (h, w, [c]).
    center_xy : tuple of float
        Target (x, y) center coordinates on the canvas (floats allowed).
    canvas_w, canvas_h : int
        Dimensions of the conceptual large canvas.

    Returns
    -------
    subimg : np.ndarray
        Image of shape (sub_h, sub_w, [c]) representing the patch rendered
        at the correct subpixel position.
    y_slice, x_slice : slice
        Slices indicating where `subimg` should be inserted into the
        larger canvas, e.g. `canvas[y_slice, x_slice] = subimg`
    """
    assert len(patch.shape) == 2 or patch.shape[-1] == 4

    patch_h, patch_w = patch.shape[:2]
    #patch_h += 4
    #patch_w += 4
    cx, cy = center_xy

    # Compute top-left corner of patch in canvas coordinates
    top_left_x = cx - patch_w / 2
    top_left_y = cy - patch_h / 2

    # Determine bounding box intersection with canvas
    x0 = int(np.floor(top_left_x))
    y0 = int(np.floor(top_left_y))
    
    #add 1 to allow for antialiasing
    x1 = x0 + patch_w + 1
    y1 = y0 + patch_h + 1
    # Clip to canvas boundaries
    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(canvas_w, x1)
    clip_y1 = min(canvas_h, y1)

    if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
        # Patch is fully outside
        return np.zeros((0, 0, *patch.shape[2:]), patch.dtype), slice(0, 0), slice(0, 0)

    # Compute affine transform for subpixel positioning
    M = np.array([
        [1, 0, top_left_x - x0],
        [0, 1, top_left_y - y0]
    ], dtype=np.float32)

    # Warp only the visible region (antialiased subpixel placement)
    region_w = x1 - x0
    region_h = y1 - y0
    warped = cv2.warpAffine(
        patch, M, (region_w, region_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Crop to the visible area
    subimg = warped[
        (clip_y0 - y0):(clip_y1 - y0),
        (clip_x0 - x0):(clip_x1 - x0)
    ]

    # Corresponding canvas slice
    y_slice = slice(clip_y0, clip_y1)
    x_slice = slice(clip_x0, clip_x1)

    return subimg, y_slice, x_slice



def draw_patch_along_polyline(canvas: np.ndarray, 
                              polyline: np.ndarray, 
                              patches: list[np.ndarray],
                              params: BrushParams, 
                              color: tuple[int, int, int] = (0, 0, 0), 
                              texture: np.ndarray | None = None,
                              size_profile: np.ndarray | None = None) -> np.ndarray:
    """
    Copies a patch with transparency along a polyline onto a grayscale image, applying various transformations.

    Args:
        image: The grayscale destination image as a NumPy array.
        polyline: A NumPy array of shape (n, 2) representing the vertices of the polyline.
        patch: A small square image with 2 channels (grayscale and alpha) to be placed along the polyline.
        params: A named tuple containing transformation parameters for the brush

    Returns:
        The resulting image with the patches drawn along the polyline.
    """
    assert params.distance_fraction > 0
    assert len(canvas.shape) > 2 and canvas.shape[-1] == 4
    assert all (len(patch.shape) == 2 for patch in patches)
    assert texture is None or len(texture.shape) == 2
        
    
    tiled = tile_image(texture, *canvas.shape[:2]) if texture is not None else None

    randstream = bounded_normal_stream(0, 0.3, -1, 1, 1000)
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if tiled is not None and params.texture_mode == 1:
        #1-channel alpha image
        output_image = np.zeros(canvas.shape[:2], dtype=np.uint8)
    else:
        # render stroke onto a fresh transparent buffer so the final opacity
        # multiplication only affects stroke alpha, not the background
        output_image = np.zeros_like(canvas)
    canvas_h, canvas_w = output_image.shape[:2]

    # tracks the local per-pixel opacity multiplier applied to the stroke at the end.
    # initialized to 255 so untouched pixels leave stroke alpha unchanged.
    opacity_layer = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # Calculate the length of each segment in the polyline
    segment_lengths = np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1))
    total_polyline_length = np.sum(segment_lengths)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    
    startphase = -1
    if params.start_phase is not None:
        startphase = params.start_phase.fraction * params.size_pixels
    endphase = total_polyline_length + 10
    if params.end_phase is not None:
        endphase = max(total_polyline_length - (params.size_pixels * params.end_phase.fraction), startphase+1)

    current_distance = 0
    while current_distance <= total_polyline_length:
        # Determine which segment the current distance falls into
        segment_index = np.searchsorted(cumulative_lengths, current_distance, side='right') - 1
        segment_index = min(segment_index, len(polyline) - 2)

        # Calculate the interpolation factor within the segment
        distance_into_segment = current_distance - cumulative_lengths[segment_index]
        if segment_lengths[segment_index] > 0:
            t = distance_into_segment / segment_lengths[segment_index]
        else:
            t = 0
        
        # Interpolate to find the point on the polyline
        start_point = polyline[segment_index]
        end_point = polyline[segment_index + 1]
        point = start_point * (1 - t) + end_point * t
        
        curr_patch = random.choice(patches)
        
        
        curr_opacity = 1.0
        size_factor = None
        if size_profile:
            u = current_distance / total_polyline_length
            size_factor = linear_interpolate(size_profile, u)
        elif current_distance < startphase and params.start_phase is not None:
            u = current_distance / startphase
            if params.start_phase.size_profile is not None:
                size_factor = linear_interpolate(params.start_phase.size_profile, u)
            if params.start_phase.opacity_profile is not None:
                curr_opacity = linear_interpolate(params.start_phase.opacity_profile, u)
        elif current_distance > endphase and params.end_phase is not None:
            u = (current_distance - endphase) / (params.end_phase.fraction * params.size_pixels)
            if params.end_phase.size_profile is not None:
                size_factor = linear_interpolate(params.end_phase.size_profile, u)
            if params.end_phase.opacity_profile is not None:
                curr_opacity = linear_interpolate(params.end_phase.opacity_profile, u)
        if size_factor is None:
            size_factor = (1 + np.random.uniform(-params.size_spread_fraction, params.size_spread_fraction))
        # a whole-brush opacity_phase takes priority over any phase opacity_profile
        if params.opacity_phase is not None and total_polyline_length > 0:
            u = min(1.0, current_distance / total_polyline_length)
            curr_opacity = linear_interpolate(params.opacity_phase.opacity_profile, u)
        curr_size = max(1.0, size_factor * params.size_pixels)
        scale = curr_size/float(curr_patch.shape[0])
        # print(size_factor, curr_size, scale)
        angle_offset_degrees = -np.random.uniform(0, 360) * params.rotation_spread_fraction
        angle_degrees = (-params.base_rotation * 360 + angle_offset_degrees) % 360
        
        # Apply random positional spreads
        #
        half_band_x = 0.5* (params.x_spread+1)*curr_size - 0.5*curr_size
        half_band_y = 0.5* (params.y_spread+1)*curr_size - 0.5*curr_size
        # print(half_band_y)
        x_offset = next(randstream) * half_band_x
        y_offset = next(randstream) * half_band_y
        
        pos_xy = point[0] + x_offset, point[1] + y_offset

        patch_roi, slice_y, slice_x = patch_transform(curr_patch, canvas_w, canvas_h, pos_xy, angle_degrees, scale)
        patch_opacity =  np.clip(params.base_patch_opacity + np.random.uniform(0, 1) * params.rotation_spread_fraction, 0, 1) * 255
        
        patch_roi = blend_masks(patch_roi, patch_opacity * np.ones_like(patch_roi))

        # patch is not fully outside image
        if min(patch_roi.shape) > 0:
            opacity_value = float(np.clip(curr_opacity * params.opacity, 0.0, 1.0))
            opacity_layer[slice_y, slice_x] = int(round(opacity_value * 255))

            if tiled is not None and params.texture_mode == 1:
                output_image[slice_y, slice_x] = add_masks(output_image[slice_y, slice_x], patch_roi)
            else:
                colorstack = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
                colorpatch = np.tile(colorstack, (*patch_roi.shape[:2], 1))
                if tiled is not None :
                    tiled_roi = tiled[slice_y, slice_x]
                    patch_roi = blend_masks(patch_roi, tiled_roi).astype(np.uint8)
                    
                colorpatch_4channel = add_alpha_to_color(colorpatch, patch_roi)
                # cv2.imwrite("patch_colored.png", tmp)
                output_image[slice_y, slice_x] = alpha_blend_4channel(colorpatch_4channel, output_image[slice_y, slice_x])

        current_distance += curr_size * params.distance_fraction
        
    if tiled is not None and params.texture_mode == 1:
        textured = blend_masks(tiled, output_image)
        textured = ((textured.astype(np.float32) * opacity_layer.astype(np.float32)) / 255.0).clip(0, 255).astype(np.uint8)
        colorstack = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        colorpatch = np.tile(colorstack, (*textured.shape[:2], 1))
        colored = add_alpha_to_color(colorpatch, textured)
        output_image = alpha_blend_4channel(colored, canvas)
    else:
        new_alpha = ((output_image[..., 3].astype(np.float32) * opacity_layer.astype(np.float32)) / 255.0).clip(0, 255).astype(np.uint8)
        output_image[..., 3] = new_alpha
        output_image = alpha_blend_4channel(output_image, canvas)

    return output_image
