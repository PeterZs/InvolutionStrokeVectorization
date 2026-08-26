import os
import random
import numpy as np
import torch
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import inspect
import wandb
import warnings
import json
from types import ModuleType
from datetime import datetime
from dataclasses import dataclass, field
import logging
from runconfig import RunConfig
import pytorch_lightning as pl
import matplotlib.colors as mcolors
import itertools
from PIL import Image
import cv2

CONFIG_PATH = "config.env"
REQUIRED_VARS = ["DATA_ROOT", "CHECKPOINT_DIR"]


    
def parse_loss_funcs(s: str) -> dict[str, float]:
    """
    Parse CLI string of format "MSELoss:1.0,L1Loss:0.5" into a dictionary
    """
    if not s:
        raise ValueError("invald string format for the loss functions")
    result = {}
    for pair in s.split(","):
        key, val = pair.split(":")
        result[key.strip()] = float(val.strip())
    return result


def set_seed(seed):
    """Set seed for reproducibility from environment variable SEED."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
    print(f"Global seed set to {seed}")

def load_env():
    """Load environment variables from a file."""
    load_dotenv(CONFIG_PATH)
    for i in REQUIRED_VARS:
        assert i in os.environ, f"{i} not found in env variables"

def get_rank():
    return int(os.environ.get("SLURM_PROCID", 0))

def prepare_system():
    warnings.filterwarnings(
        "ignore",
        message=".*'pin_memory' argument is set as true but not supported.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Consider setting `persistent_workers=True`.*",
        category=UserWarning
    )

    class FilterLitModelsTip(logging.Filter):
        def filter(self, record):
            # Return False if the message contains the specific tip to suppress it
            return "seamless cloud uploads and versioning" not in record.getMessage()
    # pl_loggers = [ logging.getLogger(name).name for name in logging.root.manager.loggerDict if 'pytorch_lightning' in name ]
    # print(pl_loggers)
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")

    # Attach the filter
    logger.addFilter(FilterLitModelsTip())
        
    if torch.cuda.is_available():
        print("clearing cuda cache")
        torch.cuda.empty_cache()
        torch.cuda.memory_summary()
        torch.set_float32_matmul_precision('medium')
        
def get_run_name_tags(config, previous_run_id):
    name = config.run_id
    tags = []
    if config.mode == 'test':
        tags.append('testset')
        name ="test_" +name + "_from_"+ previous_run_id
    elif config.mode == 'train' and previous_run_id is not None:
        # resuming from a previous run
        name += "resume_" + previous_run_id
        
    if config.debug_run:
        tags.append('debug')
        name = "debug_" + name
    return name, tags

def gather_dirs_from_names(
    data_root: str,
    names: list[str],
    split: str | None,
    split_first = True,
) -> list[str]:
    """
    Construct directory paths based on dataset names.

    Args:
        data_root (str): Base directory root.
        split (str): Data split (e.g., 'train', 'test', 'val').
        names (List[str]): List of dataset names.
        split_first (bool): If True, path is data_root/split/name.
                            If False, path is data_root/name/split.

    Returns:
        List[str]: List of constructed directory paths.
    """
    dirs = []
    for name in names:
        if split is None:
            path = os.path.join(data_root, name)
        if split_first:
            path = os.path.join(data_root, split, name)
        else:
            path = os.path.join(data_root, name, split)

        dirs.append(path)

    return dirs
    
 


    


def get_nn_module_subclasses(module):
    """
    Given a Python module, return a dict mapping class names to class objects
    for all classes in the module that are subclasses of nn.Module.
    """
    subclasses = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Make sure the class is defined in this module and is a subclass of nn.Module
        if issubclass(obj, torch.nn.Module) and obj.__module__ == module.__name__:
            subclasses[name] = obj
    return subclasses

def log(arg, **kwargs):
    
    if get_rank() == 0 and os.environ.get("WANDB_PROJECT", None):
        wandb.log(arg, **kwargs)

def wandb_id():
    wandb_run_id = None
    if wandb.run is not None:
        wandb_run_id = wandb.run.id
    return wandb_run_id

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


def color_gen(rgb = True):
    if rgb:
        return (color_to_bgr(x)[::-1] for x in itertools.cycle(COLORS))
    return (color_to_bgr(x) for x in itertools.cycle(COLORS))

def paired_color_gen(rgb = True):
    if rgb:
        return ((color_to_bgr(a)[::-1], color_to_bgr(b)[::-1]) for a, b in itertools.cycle(PAIRED_COLORS))
    
    return ((color_to_bgr(a), color_to_bgr(b)) for a, b in itertools.cycle(PAIRED_COLORS))

def to_img(img: torch.Tensor):
    """
    Convert a tensor in [0,1] float range to uint8 [0,255].
    Works for tensors from GPU.

    Args:
        img (Tensor): shape [C,H,W] or [B,C,H,W], values in [0,1]

    Returns:
        Tensor (uint8) on CPU
    """
    img_cpu = img.detach().cpu()
    img_cpu = (img_cpu * 255).clamp(0, 255).to(torch.uint8)
    return img_cpu

def to_cv_img(img: torch.Tensor):
    """
    Convert a tensor in [0,1] float range to uint8 [0,255].
    Works for tensors from GPU.

    Args:
        img (Tensor): shape [C,H,W] or [B,C,H,W], values in [0,1]

    Returns:
        Tensor (uint8) on CPU
    """
    img_cpu = img.detach().cpu()
    img_cpu = (img_cpu * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    return img_cpu.numpy()

def to_pil_img(cv_img: np.ndarray):
    return  Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def to_tensor_img(img: np.ndarray):
    img_tensor = torch.from_numpy(img).float() / 255.0
    return img_tensor.permute(2, 0, 1)


def trainer_strat(trainer: pl.Trainer):
    return trainer.strategy.__class__.__name__


class CheckParamsOnTrain(pl.Callback):
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int) -> None:
        for name, p in pl_module.model.named_parameters():
            # Check requires_grad
            assert p.requires_grad, f"Parameter '{name}' does not require gradients"

            # Check for NaN or Inf
            if not torch.isfinite(p).all():
                raise ValueError(
                    f"Parameter '{name}' contains NaN or Inf at epoch {trainer.current_epoch}"
                )
                
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        last = None
        for name, param in pl_module.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    last = name, param
        
        if last is not None:
             raise ValueError(
                    f"Parameter '{name}' contains NaN or Inf in grad"
                )
             

def save_matrix(mat: torch.Tensor, n, path: str, add_index = True):
    npmat = mat[:n, :n].detach().cpu().numpy()
    if add_index:
        tmp = np.arange(0, len(npmat))
        npmat[0] = tmp
        npmat[:, 0] = tmp
    np.savetxt(path, npmat, delimiter=",", fmt="%.3f")