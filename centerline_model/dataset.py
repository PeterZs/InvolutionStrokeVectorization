import os
from PIL import Image
from torch.utils.data import random_split, Subset, Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch

import os
from torch.utils.data import Dataset
from PIL import Image, ImageChops, ImageEnhance

from typing import NamedTuple
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import ImageFilter, Image
import numpy as np
import csv
import cv2

class NamedConcatDataset(ConcatDataset):
    """
    A ConcatDataset that keeps track of dataset names.

    Args:
        datasets (list): List of torch Dataset objects.
        names (list[str]): List of names corresponding to each dataset.
    """

    def __init__(self, datasets, names):
        if len(datasets) != len(names):
            raise ValueError(
                f"Expected the same number of datasets and names, "
                f"got {len(datasets)} datasets and {len(names)} names."
            )
        super().__init__(datasets)
        self.names = list(names)
        

class NamedDataLoader(DataLoader):
    def __init__(self, *args, dataset_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_names = dataset_names


def random_shadow1(img: Image.Image, blur_sigma: float = 30.0) -> Image.Image:
    """
    Adds a random blurred elliptical shadow to a PIL Image using the Overlay blend mode.
    Designed to be used as a custom augmentation transform for ML models.
    
    :param img: Input PIL Image
    :param blur_sigma: Constant for the Gaussian Blur intensity. Higher = softer shadow.
    :return: Augmented PIL Image
    """
    # Ensure image is RGB
    img = img.convert("RGB")
    width, height = img.size
    
    # 1. Create a neutral base for the Overlay blend mode (128 is invisible in Overlay)
    # Shape is (height, width, 3) for RGB
    shadow_layer = np.full((height, width, 3), 128, dtype=np.uint8)
    
    # 2. Generate random parameters for the ellipse
    # Center coordinates
    center_x = random.randint(int(width * 0.1), int(width * 0.9))
    center_y = random.randint(int(height * 0.1), int(height * 0.9))
    
    # Axes (width and height of the ellipse)
    axis_x = random.randint(int(width * 0.2), int(width * 0.8))
    axis_y = random.randint(int(height * 0.2), int(height * 0.8))
    
    # Rotation angle
    angle = random.randint(0, 360)
    
    # Shadow Darkness (0 is black, 100 is dark gray. Keep under 128 to ensure it darkens)
    intensity = random.randint(20, 80)
    color = (intensity, intensity, intensity)
    
    # 3. Draw the solid ellipse onto the NumPy array
    cv2.ellipse(
        shadow_layer, 
        center=(center_x, center_y), 
        axes=(axis_x, axis_y), 
        angle=angle, 
        startAngle=0, 
        endAngle=360, 
        color=color, 
        thickness=-1 # -1 means filled
    )
    
    # 4. Apply Gaussian Blur
    # We use (0,0) for kernel size so OpenCV calculates it automatically based on blur_sigma
    shadow_layer = cv2.GaussianBlur(shadow_layer, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    
    # 5. Convert back to PIL Image
    shadow_pil = Image.fromarray(shadow_layer)
    
    # 6. Apply the Overlay blend mode
    result_img = ImageChops.overlay(img, shadow_pil)
    
    return result_img


def random_shadow(img: Image.Image, blur_sigma: float = 30.0, out= "greyscale", max_rel_size=0.8) -> Image.Image:
    """
    Adds a random blurred elliptical shadow using an Alpha Mask and Overlay blend mode.
    """
    # Ensure image is RGB
    img = img.convert("RGB")
    width, height = img.size
    
    # 1. Random parameters for the ellipse
    center_x = random.randint(int(width * 0.1), int(width * 0.9))
    center_y = random.randint(int(height * 0.1), int(height * 0.9))
    axis_x = random.randint(int(width * 0.2), int(width * max_rel_size))
    axis_y = random.randint(int(height * 0.2), int(height * max_rel_size))
    angle = random.randint(0, 360)
    
    # 2. Shadow Color (Overlay darkens when RGB values are < 128)
    #intensity = random.randint(20, 80)
    intensity = 0
    shadow_color = (intensity, intensity, intensity)
    
    # Create a solid image of the shadow color
    shadow_layer = Image.new("RGB", (width, height), shadow_color)
    
    # 3. Create the Alpha Mask (Black background, White ellipse)
    # Using numpy/OpenCV for the drawing and blurring because it's much faster
    mask = np.zeros((height, width), dtype=np.uint8)
    
    cv2.ellipse(
        mask, 
        center=(center_x, center_y), 
        axes=(axis_x, axis_y), 
        angle=angle, 
        startAngle=0, 
        endAngle=360, 
        color=255,   # 255 (White) means fully opaque shadow
        thickness=-1 # -1 means filled
    )
    
    # 4. Apply Gaussian Blur to the Alpha Mask
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    
    # Convert mask to PIL Image (mode "L" for grayscale/luminance)
    mask_pil = Image.fromarray(blurred_mask, mode="L")
    
    # 5. Apply the Overlay blend mode 
    # This creates a version of the image where the whole thing has the shadow applied
    overlaid_img = ImageChops.overlay(img, shadow_layer)
    overlaid_img = ImageChops.blend(img, overlaid_img , 0.9)
    overlaid_img = ImageChops.blend(overlaid_img, shadow_layer, 0.2)
    
    # 6. Composite the images using the blurred Alpha Mask
    # Where mask is white (255), it uses overlaid_img. Where black (0), it uses the original img.
    # The blurred edges create a perfect, smooth gradient between the two.
    final_img = Image.composite(overlaid_img, img, mask_pil)
    if out== "greyscale":
        final_img = final_img.convert("L")
    return final_img


def add_noise_pattern(
    img,
    sigma_fine=0.2,
    sigma_coarse=2.0,
    coarse_weight=0.75,
    zero_quantile=0.25,
    skew_exponent=2.0,
    target_mean=5.0,
    max_value=20.0,
):
    """Additively overlay a fine, near-black speckle pattern onto an image.

    The pattern is generated at the input's spatial size (last two dims) and
    added, broadcasting across any leading channel/batch dims. Intensity
    defaults are in 0-255 units; scale `target_mean` and `max_value` to match
    your image's range.

    img: float tensor of shape (..., H, W)
    returns: img + noise, same shape/dtype/device
    """
    *_, H, W = img.shape

    # white noise -> two-scale Gaussian blur (hop to numpy/cv2; non-differentiable)
    w = torch.randn(H, W).numpy().astype(np.float32)
    n = ((1.0 - coarse_weight) * cv2.GaussianBlur(w, (0, 0), sigma_fine)
         + coarse_weight * cv2.GaussianBlur(w, (0, 0), sigma_coarse))
    n = torch.from_numpy(n).to(device=img.device, dtype=torch.float32)
    n = n / (n.std() + 1e-12)

    # skew + clip toward black, match brightness, apply ceiling
    x = torch.clamp(n - torch.quantile(n, zero_quantile), min=0.0) ** skew_exponent
    x = x / (x.mean() + 1e-12) * target_mean
    noise = torch.clamp(x, 0.0, max_value)

    return img + noise.to(img.dtype)


class AugmentationConfig(NamedTuple):
    # Spatial
    rotate: bool = True
    flip: bool = True
    translate: bool = False
    max_translate: int = 15

    # Cropping
    random_crop: bool = True
    region_height: int = 800
    region_width: int = 800

    # Grayscale conversion (deterministic, applied BEFORE augmentations)
    input_grayscale: bool = True
    output_grayscale: bool = True

    # Noise
    gaussian_noise_prob: float = 0.25
    noise_std_min: float = 0.075
    noise_std_max: float = 0.125

    # Pattern noise (mutually exclusive with gaussian noise)
    pattern_noise_prob: float = 0.3
    pattern_sigma_fine_min: float = 0.35
    pattern_sigma_fine_max: float = 0.45
    # pattern_coarse_weight_min: float = 0.7
    # pattern_coarse_weight_max: float = 0.85
    pattern_zero_quantile_min: float = 0.15
    pattern_zero_quantile_max: float = 0.35
    pattern_skew_exponent_min: float = 1.75
    pattern_skew_exponent_max: float = 2.25
    # target_mean/max_value scaled from 0-255 defaults (5, 20) to [0,1] range
    pattern_target_mean_min: float = 4.0 /255.0
    pattern_target_mean_max: float = 6.0/255.0
    pattern_max_value_min: float = 10.0/255.0
    pattern_max_value_max: float = 25.0/255.0

    salt_pepper_prob: float = 0.5
    sp_amount: float = 0.01
    
    brightness_prob: float = 0.3
    brightness_min: float = 0.11
    brightness_max: float = 0.18

    # Blur (applied after crop, BEFORE to_tensor)
    blur_prob: float = 0.15
    blur_sigma_min: float = 0.8
    blur_sigma_max: float = 1.75
    
    #add shadows
    shadow_prob = 0.9
    shadow_blur_min: int = 20
    shadow_blur_max: int  = 65

class PairedAugment:
    """
    Paired augmentations for image-to-image tasks.
    - Spatial transforms (same for input & output): rotations, flips, translations, random crop.
    - Gaussian blur: input-only, applied after crop and before to_tensor, with probability.
    - Noise: input-only, applied after to_tensor.
    - Deterministic grayscale conversion options for input and/or output (applied first).
    """

    def __init__(self, cfg: AugmentationConfig, invert = True, test_mode = False):
        self.cfg = cfg
        self.gray = transforms.Grayscale(num_output_channels=1)
        self.invert = invert
        self.test_mode = test_mode


    def _apply_pil_blur(self, pil_img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.cfg.blur_sigma_min, self.cfg.blur_sigma_max)
        return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    def _add_gaussian_noise_pil(self, img: Image.Image) -> Image.Image:
        """
        Adds Gaussian noise to a PIL Image.
        """
        # 1. Convert PIL image to a float32 NumPy array
        # We use float32 to prevent overflow/underflow when adding negative/large noise values
        img_array = np.array(img, dtype=np.float32)
        
        # 2. Generate Gaussian noise with the same shape as the image
        std = random.uniform(self.cfg.noise_std_min, self.cfg.noise_std_max)
        noise = np.random.normal(loc=0.0, scale=std * 255, size=img_array.shape)
        
        # 3. Add noise, clip values to valid 8-bit range, and convert to uint8
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # 4. Convert back to PIL Image
        return Image.fromarray(noisy_array)

    def _add_gaussian_noise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.cfg.noise_std_min, self.cfg.noise_std_max)
        return x + torch.randn_like(x) * std

    def _add_pattern_noise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return add_noise_pattern(
            x,
            sigma_fine=random.uniform(cfg.pattern_sigma_fine_min, cfg.pattern_sigma_fine_max),
            sigma_coarse=2,
            coarse_weight=0.75,
            zero_quantile=random.uniform(cfg.pattern_zero_quantile_min, cfg.pattern_zero_quantile_max),
            skew_exponent=random.uniform(cfg.pattern_skew_exponent_min, cfg.pattern_skew_exponent_max),
            target_mean=random.uniform(cfg.pattern_target_mean_min, cfg.pattern_target_mean_max),
            max_value=random.uniform(cfg.pattern_max_value_min, cfg.pattern_max_value_max),
        )

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

    def _random_crop_pil(self, inp: Image.Image, out: Image.Image):
        w, h = inp.size
        rh = self.cfg.region_height
        rw = self.cfg.region_width

        if w < rw or h < rh:
            # skip crop if region is larger than image (could resize elsewhere)
            return inp, out

        x0 = random.randint(0, w - rw)
        y0 = random.randint(0, h - rh)
        inp_cropped = inp.crop((x0, y0, rw+x0, rh+y0))
        out_cropped = out.crop((x0, y0, rw+x0, rh+y0))
        assert inp_cropped.height == out_cropped.height
        assert inp_cropped.width == out_cropped.width
        return inp_cropped, out_cropped

    def __call__(self, inp: Image.Image, out: Image.Image):
        cfg = self.cfg

        if cfg.input_grayscale:
            inp = self.gray(inp)
        if cfg.output_grayscale:
            out = self.gray(out)

        if self.test_mode:
            inp_t = F.to_tensor(inp)
            out_t = F.to_tensor(out)
            mask = (inp_t != 1).float().clone().detach()
            if self.invert:
                inp_t = 1 - inp_t
                out_t = 1 - out_t
                # increase output brightness due to way it was rasterized
                out_t *= 2.4
            return inp_t, out_t, mask

        # Random crop (keep region fully inside image)
        if cfg.random_crop:
            inp, out = self._random_crop_pil(inp, out)

        # Gaussian blur on PIL input only (apply after crop, before to_tensor)
        was_blurred = False
        if random.random() < cfg.blur_prob:
            inp = self._apply_pil_blur(inp)
            was_blurred = True
        #we don't want blurring and reducing brightness to co-ocur (then the strokes are really too dim)
        elif random.random() < cfg.brightness_prob:
            mult = random.uniform(cfg.brightness_min, cfg.brightness_max)
            # Reduce brightness of the (eventually) inverted image, before the
            # shadow is added. The image here is still a non-inverted PIL image,
            # so apply the equivalent point op: an inverted pixel has value
            # v = 255 - p, we want mult * v, i.e. p -> 255 - mult * (255 - p).
            offset = 255.0 * (1.0 - mult)
            inp = inp.point(lambda p: int(offset + mult * p))
        
        if random.random() < cfg.shadow_prob:
            def lift_shadows(p):
                # Gamma-style lift (gamma < 1 brightens shadows)
                gamma = 0.15
                return int(255 * ((p / 255) ** gamma))

            # Apply curve to all channels
            #inp = inp.point(lift_shadows)
            
            sig = random.randint(cfg.shadow_blur_min, cfg.shadow_blur_max)
            inp = random_shadow(inp, blur_sigma=sig, max_rel_size=0.5)
        
            # if cfg.add_gaussian_noise:
            #     inp = self._add_gaussian_noise_pil(inp)

        # ---------- Convert to tensor ----------
        inp_t = F.to_tensor(inp)   # CHW, float32, [0,1]
        out_t = F.to_tensor(out)
        assert inp_t.shape[0] == 1
        # ---------- Spatial transforms ----------
        # Random 90° rotation
        if cfg.rotate:
            k = random.randint(0, 3)
            if k:
                angle = 90 * k
                inp_t = F.rotate(inp_t, angle)
                out_t = F.rotate(out_t, angle)

        # Random flips
        if cfg.flip:
            if random.random() < 0.5:
                inp_t = F.hflip(inp_t)
                out_t = F.hflip(out_t)
            if random.random() < 0.5:
                inp_t = F.vflip(inp_t)
                out_t = F.vflip(out_t)


        mask = (inp_t != 1).float().clone().detach()

        inp_t = inp_t.clamp(0.0, 1.0)
        out_t = out_t.clamp(0.0, 1.0)

        if self.invert:
            inp_t = 1- inp_t
            out_t = 1-out_t
            # increase output brightness due to way it was rasterized
            out_t *= 2.4

        
        # ---------- Input-only noise (tensor ops) ----------
        # Applied AFTER invert so the background is dark (0). Pattern noise is
        # strictly positive — adding it to a white (1.0) background would just
        # be clamped away.
        # Gaussian noise and pattern noise are mutually exclusive: a single draw
        # picks at most one of them.
        gauss_p = cfg.gaussian_noise_prob
        pattern_p = cfg.pattern_noise_prob
        if gauss_p + pattern_p > 0:
            r = random.random()
            if r < gauss_p:
                inp_t = self._add_gaussian_noise_tensor(inp_t)
            elif r < gauss_p + pattern_p:
                inp_t = self._add_pattern_noise_tensor(inp_t)

        if random.random() < cfg.salt_pepper_prob:
            inp_t = self._salt_and_pepper_tensor(inp_t)

        inp_t = inp_t.clamp(0.0, 1.0)
        return inp_t, out_t, mask

class ImageToImageDataset(Dataset):
    """
    Dataset for image-to-image tasks using a normal directory.

    Rules:
      - Output image: filename_no_ext ends with "_output"
      - Input image: all others
      - Match by prefix = filename_no_ext with the last "_suffix" removed.
        (For output: remove "_output"; for input: use first two underscore components)
    """

    def __init__(self, dir_path, csv_pairs_path, augmentation: PairedAugment):
        self.dir_path = dir_path
        self.csv_pairs_path = csv_pairs_path
        self.augmentation = augmentation

        self.pairs = self._load_pairs()

    def _load_pairs(self):

        pairs = []
        with open(self.csv_pairs_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    pairs.append((row[0], row[1]))
        print(f"{self.dir_path}: loaded {len(pairs)} pairs from cache")
        return pairs


    def __len__(self):
        return len(self.pairs)

    def _open_image(self, filename):
        path = os.path.join(self.dir_path, filename)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        input_name, output_name = self.pairs[idx]

        input_img = self._open_image(input_name)
        output_img = self._open_image(output_name)

        input_img, output_img, mask = self.augmentation(input_img, output_img)

        base_name = os.path.basename(input_name).rsplit(".", 1)[0]
        return input_img, output_img, mask, base_name



class InvertTensor:
    def __call__(self, x):
        return 1.0 - x

def get_dataloaders(
    dataset_dict: dict[str, str],
    image_size: int,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    val_split=0.1,
    tiny_set : int = -1,  # new option
    seed = 42,
    test_mode = False
):
    """
    Returns train and validation DataLoaders, with optional overfitting mode.

    Args:
        dataset_dirs (list[str]): dataset folders
        batch_size (int)
        shuffle (bool)
        num_workers (int)
        val_split (float): fraction of dataset used for validation
        tiny_set (int or None): if >0, only use tiny_set samples. 
        This results in no validation dataset, and batch_size = 1. Used for debugging.
    """

    cfg = AugmentationConfig(region_height=image_size, region_width=image_size)
   
    augm = PairedAugment(cfg, test_mode=test_mode)
    

    # Load datasets
    datasets = []
    for name, dir in dataset_dict.items():
        pairs_csv_dir = dir + ".csv"
        print(dir)
        print(pairs_csv_dir)
        assert os.path.isfile(pairs_csv_dir)
        ds = ImageToImageDataset(
            dir_path=dir,
            csv_pairs_path=pairs_csv_dir,
            augmentation=augm
        )
        datasets.append(ds)

    # Combine datasets if multiple
    full_dataset = ConcatDataset(datasets)
    print("concatenated dataset")
    # Limit dataset to a tiny number of samples (useful for debugging)
    if tiny_set >0:
        overfit_samples = min(tiny_set, len(full_dataset))
        full_dataset = Subset(full_dataset, list(range(overfit_samples)))

    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    print("train size is", train_size)

    #Reproducible dataloader behavior
    g = torch.Generator()
    g.manual_seed(seed)

    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_loader = NamedDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
            dataset_names=list(dataset_dict.keys())
            #persistent_workers=True #this causes problems in distributed training
        )
    else:
        train_dataset = full_dataset
        val_loader = None
    
    print("init train_loader")
    train_loader = NamedDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        dataset_names=list(dataset_dict.keys())
        #persistent_workers=True #this causes problems in distributed training
    )

    return train_loader, val_loader