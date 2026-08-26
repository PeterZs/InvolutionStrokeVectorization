import os
from PIL import Image
from torch.utils.data import random_split, Subset, Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch
import json
from skimage.draw import line
from scipy.ndimage import gaussian_filter1d

import cv2

import os
from torch.utils.data import Dataset
from PIL import Image

from typing import NamedTuple
import random
import torch
from typing import NamedTuple
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
from PIL import ImageFilter, Image
import numpy as np
import csv
import utils

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from scipy.interpolate import interp1d

TARGET_SIZE = 512

def compute_matrix_for_rescaling(in_w: float, in_h: float, target: float, align: str = "center") -> np.ndarray:
    """
    Compute matrix (a, b, c, d, e, f) for uniform scaling to fit (no stretching)
    into a square of size `target`. `align` can be "center" or "top-left".
    - a = scale_x = scale
    - d = scale_y = scale
    - b = c = 0
    - e = tx, f = ty (translation)
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input width and height must be positive numbers")

    s = min(target / in_w, target / in_h)
    scaled_w = in_w * s
    scaled_h = in_h * s

    if align == "center":
        tx = (target - scaled_w) / 2.0
        ty = (target - scaled_h) / 2.0
    elif align == "top-left":
        tx = 0.0
        ty = 0.0
    else:
        raise ValueError("invalid align")

    # matrix(a b c d e f) with no skew/rotation
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def apply_transform(matrix: np.ndarray, points: np.ndarray):
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack([points, ones])
    transformed = hom @ matrix.T
    return transformed[:, :2]

def resample_polyline(points, N):
    """resamples a single polyline of shape (V, 2) to (N, 2). """
    points = np.asarray(points)
    if len(points) == 1:
        return np.repeat(points, N, axis=0)

    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    
    total_length = cum_dists[-1]
    if total_length == 0:
        return np.repeat(points[:1], N, axis=0)
    
    cum_dists_norm = cum_dists / total_length
    interpolator = interp1d(cum_dists_norm, points, axis=0, kind='linear')
    t = np.linspace(0, 1, N)
    resampled = interpolator(t)
    
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled


def rescale_polylines(lines: dict[int, np.ndarray], in_w, in_h, target):
    mat = compute_matrix_for_rescaling(in_w, in_h, target)
    return {k: apply_transform(mat, v) for k, v in lines.items()}
    

def draw_polyline_cv2(image, polyline, color, aa=False):
    # 1. Ensure the polyline is in the correct format (int32)
    # OpenCV requires a list of arrays with shape (number_of_points, 1, 2)
    pts = polyline.astype(np.int32).reshape((-1, 1, 2))
    
    # 2. Draw the polyline
    # lineType=cv2.LINE_8 ensures no anti-aliasing
    # isClosed=False ensures it's an open path (polyline) rather than a polygon
    cv2.polylines(
        image, 
        [pts], 
        isClosed=False, 
        color=color, 
        thickness=1, 
        lineType=cv2.LINE_AA if aa else cv2.LINE_8
    )


def get_polyline_pixels(polyline, dim=None):
    """
    Returns the ordered (N, 2) array of [y, x] pixel coordinates for a polyline.
    
    Args:
        polyline: np.array of shape (N, 2) containing [x, y] coordinates.
        dim: Tuple (height, width) to clip pixels. Pixels outside are removed.
             Order is preserved.
    """
    all_segments = []
    
    for i in range(len(polyline) - 1):
        p0 = polyline[i]
        p1 = polyline[i+1]
        
        # 1. Generate line pixels for this segment
        # skimage uses (row, col) which is (y, x)
        rr, cc = line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
        segment = np.column_stack((rr, cc)) # shape (M, 2) as [x, y]
        
        # 2. Filter by dimensions if provided
        if dim is not None:
            h, w = dim
            # mask: 0 <= x < width AND 0 <= y < height
            mask = (segment[:, 0] >= 0) & (segment[:, 0] < w) & \
                   (segment[:, 1] >= 0) & (segment[:, 1] < h)
            segment = segment[mask]
        
        if len(segment) > 0:
            all_segments.append(segment)

    if not all_segments:
        return np.empty((0, 2), dtype=np.int32)

    # 3. Concatenate all valid segments
    pixels = np.concatenate(all_segments, axis=0)
    
    # 4. Remove consecutive duplicates (vertices shared by segments)
    # This ensures a clean 1px stroke order without "double-stepping" on joints
    if len(pixels) > 1:
        # Keep pixel if it is different from the previous pixel
        diff_mask = np.ones(len(pixels), dtype=bool)
        diff_mask[1:] = np.any(pixels[1:] != pixels[:-1], axis=1)
        pixels = pixels[diff_mask]
    
        
    return pixels

def parabolic_profile(t):
        return 4 * t * (1 - t)
    
def noise_2d(N, sigma):
    raw_noise = np.random.randn(N, 2)
    smoothed_noise = gaussian_filter1d(raw_noise, sigma=sigma, axis=0)

    smoothed_noise = smoothed_noise / np.std(smoothed_noise)
    return smoothed_noise

def add_profile_noise(polyline: np.ndarray, profile_func = parabolic_profile, max_noise_std: float = 1.0, sigma: float = 5.0) -> np.ndarray:
    """
    Adds noise to an (N, 2) polyline based on a programmable intensity profile.
    
    :param polyline: (N, 2) numpy array representing the polyline.
    :param profile_func: A function that takes an array of floats and returns an array of scales.
    :param max_noise_std: Maximum standard deviation of the Gaussian noise.
    :return: A new (N, 2) numpy array with the applied noise.
    """
    N = polyline.shape[0]
    
    # 1. Parameterize the polyline indices from 0.0 to 1.0
    t = np.linspace(0, 1, N)
    
    # 2. Sample the noise intensity profile
    # Reshape to (N, 1) so it broadcasts correctly against the (N, 2) noise array
    intensity_profile = profile_func(t).reshape(-1, 1)
    
    # 3. Generate raw 2D Gaussian noise (mean=0, std=1)
    noise = noise_2d(N, sigma=sigma)
    
    # 4. Scale the raw noise by our profile and the max noise intensity
    scaled_noise = noise * intensity_profile * max_noise_std
    
    # 5. Add noise to the original polyline
    noisy_polyline = polyline + scaled_noise
    
    # 6. Explicitly preserve endpoints to prevent floating-point precision issues
    noisy_polyline[0] = polyline[0]
    noisy_polyline[-1] = polyline[-1]
    
    return noisy_polyline

def count(seg_img: torch.Tensor):
    return len(torch.unique(seg_img)) 

def verify_segmentation(seg_img: torch.Tensor, N: int):
    c = count(seg_img)-1
    if c != N:
        print("counted on img", c)
        print("actual", N)
    return c == N
    

class AugmentationConfig(NamedTuple):
    # Yes/No Decisions
    use_blur: bool = True
    use_rotation: bool = True
    use_hflip: bool = True
    use_vflip: bool = True
    use_gaussian_noise: bool = True
    use_salt_pepper: bool = True
    
    # Probabilities
    blur_prob: float = 0.5
    rotation_prob: float = 0.5
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    noise_prob: float = 0.9
    sp_prob: float = 0.5
    
    # Magnitudes / Parameters
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.9
    noise_std: float = 0.1
    sp_amount: float = 0.03


class PairedAugment:
    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def _apply_pil_blur(self, pil_img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.cfg.blur_sigma_min, self.cfg.blur_sigma_max)
        return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _add_gaussian_noise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.cfg.noise_std

    def _salt_and_pepper_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # x is CHW in [0,1]
        c, h, w = x.shape
        mask = torch.rand(h, w, device=x.device)

        x = x.clone()
        salt = (mask < (self.cfg.sp_amount / 2.0))
        pepper = ((mask >= (self.cfg.sp_amount / 2.0)) & (mask < self.cfg.sp_amount))

        # apply channel-wise
        x[:, salt] = 1.0
        x[:, pepper] = 0.0
        return x

    def __call__(self, inp: Image.Image, seg: torch.Tensor):
        cfg = self.cfg
        
        # 1. Blur (Applied first, only on input PIL image)
        if cfg.use_blur and random.random() < cfg.blur_prob:
            inp = self._apply_pil_blur(inp)
            
        # Convert input PIL image to Tensor CHW.
        # This makes applying identical geometric transformations to both 
        # the image and the mask easy and synchronized.
        inp_t = TF.to_tensor(inp)
        assert inp_t.shape[0] == 3

        # 2. Random 90° multiples rotation (applied to both)
        if cfg.use_rotation and random.random() < cfg.rotation_prob:
            # Random multiple of 90 degrees: 1=90°, 2=180°, 3=270°
            k = random.randint(1, 3) 
            
            # Using torch.rot90 on the spatial dimensions (last two: H, W) is ideal
            # for segmentation indices because it involves zero interpolation.
            inp_t = torch.rot90(inp_t, k, dims=(1, 2))
            seg = torch.rot90(seg, k, dims=(0, 1))

        # 3. Random Flips (applied to both)
        if cfg.use_hflip and random.random() < cfg.hflip_prob:
            inp_t = TF.hflip(inp_t)
            seg = TF.hflip(seg)
            
        if cfg.use_vflip and random.random() < cfg.vflip_prob:
            inp_t = TF.vflip(inp_t)
            seg = TF.vflip(seg)
        # 4. Gaussian Noise (applied only to input tensor)
        if cfg.use_gaussian_noise and random.random() < cfg.noise_prob:
            inp_t = self._add_gaussian_noise_tensor(inp_t)

        # 5. Salt and Pepper Noise (applied only to input tensor)
        if cfg.use_salt_pepper and random.random() < cfg.sp_prob:
            inp_t = self._salt_and_pepper_tensor(inp_t)

        # Ensure image values stay valid in after additive noise
        inp_t = torch.clamp(inp_t, 0.0, 1.0)

        return inp_t, seg

class NamedDataLoader(DataLoader):
    def __init__(self, *args, dataset_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_names = dataset_names

class SceneGraphDataset(Dataset):
    """
    Dataset for Object Relation tasks.
    
    Structure per dataset_dir:
      - img/          : Greyscale images (.webp)
      - segmentation/ : RGB masks (.png or similar), color-coded IDs
      - mat/          : CSV files for relation matrix (same filename as img)
      
    Logic:
      - Mask is converted from RGB to single channel Int32 (R + G*256 + B*256^2).
      - Matrix CSV: 
          - 2: Correct relation (Ground Truth)
          - 1: Valid relation input
          - 0: Invalid
      - Input Matrix (Constraint): values > 0
      - Target Matrix (GT): values == 2
    """

    def __init__(self, root_dir, max_objects=1000):
        self.root_dir = root_dir
        self.mat_dir = os.path.join(root_dir, "mat")
        self.pairs_dir = os.path.join(root_dir, "half_edges_pairs")
        self.lines_dir = os.path.join(root_dir, "lines_json")

        self.samples = self._find_samples()
        print(f"{self.root_dir}: loaded {len(self.samples)} samples")
        self.max_objects = max_objects
        
        cfg = AugmentationConfig()
        self.augmentation = PairedAugment(cfg)

    def _find_samples(self):
        """
        Matches files across directories based on filename (ignoring extension).
        Uses 'lines_json' as the source of truth for file existence.
        """
        samples = []
        
        if not all(os.path.isdir(d) for d in [self.mat_dir, self.lines_dir]):
            raise RuntimeError(f"Directory structure missing in {self.root_dir}")

        lines_files = sorted([f for f in os.listdir(self.lines_dir) if f.endswith(".json")])

        for lines_file in lines_files:
            basename = os.path.splitext(lines_file)[0]
            
            mat_path = os.path.join(self.mat_dir, basename + ".csv")
            pair_path = os.path.join(self.pairs_dir, basename + ".csv")
            lines_path = os.path.join(self.lines_dir, lines_file)

            if os.path.exists(mat_path):
                samples.append({
                    "basename": basename,
                    "mat": mat_path,
                    "pairs": pair_path,
                    "lines": lines_path
                })
            else:
                print(f"Warning: Missing mat for {basename} in {self.root_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, path):
        """
        Converts RGB mask to Int32 ID map.
        ID = R + (G * 256) + (B * 256^2)
        """
        # Open as RGB
        mask_img = Image.open(path).convert("RGB")
        mask_np = np.array(mask_img, dtype=np.int32) # (H, W, 3)

        # Calculate polynomial encoding
        # Shape becomes (H, W)
        id_map = mask_np[:, :, 0] + \
                 (mask_np[:, :, 1] * 256) + \
                 (mask_np[:, :, 2] * 65536)
        # note that in `mask_np`, 0 is black (background).
        # so we need to subtract 1 if we want the object indexes to start at 0.
        # however, this would set the background to -1
        # so instead, we just keep the background.
        return torch.from_numpy(id_map).long() # Int64/LongTensor for PyTorch indices

    def _pad_and_create_mask(self, matrix, current_n, offset = 0):
        """
        Pads a (N, N) matrix to (max_objects, max_objects).
        Returns:
            padded_matrix: (max_objects, max_objects)
            valid_mask: (max_objects) -> 1 for real objects, 0 for pad
        """
        K = self.max_objects
        
        # Safety check: if current_n > max_objects, we must truncate
        if current_n+offset >= K:
            raise ValueError(f"Truncating object count from {current_n} to {K}")


        # 1. Create Padded Matrix
        padded_matrix = torch.zeros((K, K), dtype=torch.float32)
        padded_matrix[offset:offset+current_n, offset:offset+current_n] = matrix

        assert padded_matrix[:offset].sum() ==0.0
        assert padded_matrix[:, :offset].sum() ==0.0

        return padded_matrix

    def _load_matrix(self, path):
        """
        Parses CSV into Tensor and splits into Input/Target.
        """
        matrix_rows = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Convert strings to ints
                matrix_rows.append([int(x) for x in row])
        
        # Raw matrix: 0=Invalid, 1=Valid, 2=GroundTruth
        raw_mat = torch.tensor(matrix_rows, dtype=torch.float32)

        # Input Constraint: Is this relation allowed? (1 or 2 -> 1)
        constraint_mask = (raw_mat > 0).float()

        # Target: Is this the correct relation? (2 -> 1, others 0)
        target_matrix = (raw_mat == 2).float()

        return constraint_mask, target_matrix
    
    def _load_pairs(self, path, N, offset=0):
        result = [0] * N
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for k, v in reader:
                result[int(k)+offset] = int(v)+offset
        return torch.tensor(result)
    


    def _load_lines(self, path: str, offset: int) -> dict[int, np.ndarray]:

        with open(path, "r") as f:
            data = json.load(f)

        lines: dict[int, np.ndarray] = {}

        for key_str, polyline in data.items():
            try:
                key = int(key_str) + offset
            except ValueError:
                raise ValueError(f"Invalid key '{key_str}' in {path}, expected int.")

            arr = np.asarray(polyline, dtype=np.float64)

            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"Polyline for key {key_str} must be of shape (N, 2), "
                    f"got {arr.shape}"
                )

            lines[key] = arr

        return lines
    
    def _compute_images(self, polylines: dict, target_size):
         # step 2: build segmentation image
         
        seg_img = np.zeros((target_size, target_size), dtype=np.int32)
    
        for idx, pl in polylines.items():
            coords = get_polyline_pixels(pl, (target_size, target_size))
            seg_img[coords[:, 0], coords[:, 1]] = idx
            
        # step 3: line drawing
        img_dr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        for pl in polylines.values():
            draw_polyline_cv2(img_dr, pl, (255, 255, 255), aa=True)
        
        
        return utils.to_pil_img(img_dr), torch.from_numpy(seg_img)
    
    def _resampled_polylines(self, polylines: dict):
        
        res = torch.zeros(self.max_objects, 100, 2)
        for idx, pl in polylines.items():
            res[idx] = torch.from_numpy(resample_polyline(pl, 100))
        return res

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Image 
        # img = Image.open(sample["img"]).convert("RGB")
        # img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        # img_tensor = img_tensor.permute(2, 0, 1)
        # assert img_tensor.ndim == 3
        # assert img_tensor.shape[0] == 3

        # # 2. Load Segmentation (Int32 Map)
        # seg_tensor = self._load_mask(sample["seg"]) # [H, W]

        # 3. Load Matrices
        constraint_mask, target_matrix = self._load_matrix(sample["mat"])
        
        current_n = constraint_mask.shape[0]
        
        #IMPORTANT: since index 0 is the background, add a row and col at the beginning
        pad_constraint = self._pad_and_create_mask(constraint_mask, current_n, offset=1)
        pad_target = self._pad_and_create_mask(target_matrix, current_n, offset=1)
        pad_target[0, 0] = 1
        pad_constraint[0, 0] = 1
        
        
        # 5. Load pair info (not strictly needed for training)
        half_edge_pairs_padded = self._load_pairs(sample["pairs"], self.max_objects, offset=1)
        assert len(half_edge_pairs_padded) == pad_target.shape[0]
        
        assert torch.equal(pad_target, pad_target.T)
        assert torch.equal(pad_constraint, pad_constraint.T)
        
        # 4. Load half-edge polylines
        lines_inp = self._load_lines(sample["lines"], offset=1)
        lines_inp = rescale_polylines(lines_inp, 500, 500, TARGET_SIZE)
        tries = [(0.5, 5), (0, 5)]
        for max_noise_std, sigma in tries:
            lines = {k: add_profile_noise(v, max_noise_std=max_noise_std, sigma=sigma) for k, v in lines_inp.items()}
            #generate augmented images from lines
            img, seg_tensor = self._compute_images(lines, TARGET_SIZE)
            img_tensor, seg_tensor = self.augmentation(img, seg_tensor)
            if verify_segmentation(seg_tensor, current_n):
                break
        
        #needed for scatter() to work correctly
        seg_tensor = seg_tensor.to(torch.int64)
        #print(img_tensor.shape)
        #print(seg_tensor.shape)
        
        pl_tensor = self._resampled_polylines(lines_inp)
        
        

        return {
            "img": img_tensor, 
            "seg": seg_tensor, 
            "lines": pl_tensor,
            "R": pad_constraint, 
            "target_mat": pad_target, 
            #add 1 to include background
            "n_objects": current_n+1, 
            "pairs": half_edge_pairs_padded,
            #"lines": lines,
            "basename": sample["basename"]
        }


def get_dataloaders(
    dataset_dict: dict[str, str],
    batch_size=16,
    shuffle=True,
    num_workers=4,
    val_split=0.1,
    tiny_set : int = -1,
    seed = 42,
    max_objects = 1000
):
    """
    Returns train and validation DataLoaders.

    Args:
        dataset_dict (dict[str, str]): Key is dataset name, Value is ROOT path containing img/, segmentation/, mat/
    """

    # Load datasets
    datasets = []
    for name, root_dir in dataset_dict.items():
        print(f"Loading dataset: {name} from {root_dir}")
        if os.path.isdir(root_dir):
            ds = SceneGraphDataset(root_dir=root_dir, max_objects=max_objects)
            datasets.append(ds)
        else:
            raise ValueError(f"Path {root_dir} not found.")

    if not datasets:
        raise ValueError("No valid datasets loaded.")

    # Combine datasets
    full_dataset = ConcatDataset(datasets)
    print("Concatenated dataset size:", len(full_dataset))

    # Debugging: tiny set override
    if tiny_set > 0:
        overfit_samples = min(tiny_set, len(full_dataset))
        full_dataset = Subset(full_dataset, list(range(overfit_samples)))
        print(f"Tiny set mode: limited to {overfit_samples} samples.")

    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    print(f"Split: Train={train_size}, Val={val_size}")

    # Reproducible dataloader behavior
    g = torch.Generator()
    g.manual_seed(seed)

    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)
        val_loader = NamedDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
            dataset_names=list(dataset_dict.keys())
        )
    else:
        train_dataset = full_dataset
        val_loader = None
    
    print("Initializing train_loader...")
    train_loader = NamedDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        dataset_names=list(dataset_dict.keys())
    )
    return train_loader, val_loader