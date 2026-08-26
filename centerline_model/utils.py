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


CONFIG_PATH = "config.env"
REQUIRED_VARS = ["SCRATCH_DIR", "DATA_PATH", "CHECKPOINT_DIR"]


    
def parse_loss_funcs(s: str) -> dict[str, float]:
    """
    Parse CLI string of format "MSELoss:1.0,L1Loss:0.5" into a dictionary
    """
    if not s:
        return {"MSELoss": 1.0}
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
        
def get_run_name_tags(config, previous_state_dict):
    name = config.run_id
    tags = []
    if config.mode == 'test':
        tags.append('testset')
        name ="test_" +name + "_from_"+ previous_state_dict['run_id']
    elif config.mode == 'train' and previous_state_dict['run_id'] is not None:
        # resuming from a previous run
        name += "resume_" + previous_state_dict['run_id']
        
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
    
 
def copy_and_unpack_tar_data(data_path_str, scratch_dir_str, ignore_existing=False):
    """Copy tar data to scratch directory and extract it.

    Args:
        data_path (str or Path): Path to the .tar file.

    Returns:
        tuple: (scratch_dir: Path, data_path: Path)
    """
    data_path = Path(data_path_str)
    scratch_dir = Path(scratch_dir_str)
    
    dest = scratch_dir / data_path.stem
    if dest.exists() and not ignore_existing:
        if any(dest.glob("*.webp")):
            print("data already copied")
            return str(dest)
        else:
            raise RuntimeError(f"{dest} may be corrupted")
    
    #the train/test folders may not have been created
    os.makedirs(scratch_dir, exist_ok=True)

    # Step 1: Copy tar file to scratch
    t0 = time.perf_counter()
    copy_cmd = ["cp", "-R", str(data_path), str(scratch_dir)]
    proc = subprocess.run(copy_cmd, check=True)
    t1 = time.perf_counter()
    print(f"Copied the .tar file in {t1 - t0:0.4f} seconds", flush=True)
    # Step 2: Extract tar file
    tar_file_in_scratch = scratch_dir / data_path.name
    os.makedirs(dest, exist_ok=True)
    tar_cmd = ["tar", "-xf", str(tar_file_in_scratch), "-C", str(dest)]
    print(" ".join(tar_cmd))
    proc = subprocess.run(tar_cmd, check=True)
    t2 = time.perf_counter()
    os.remove(tar_file_in_scratch)
    print(f"Extracted the .tar file in {t2 - t1:0.4f} seconds", flush=True)
    print("successfully created '", str(dest), "'")

    return str(dest)


def copy_tar(data_path_str, scratch_dir_str, ignore_existing=False):
    data_path = Path(data_path_str)
    scratch_dir = Path(scratch_dir_str)
    dest = scratch_dir / data_path.name
    if dest.exists() and not ignore_existing:
        print("data already copied")
        return str(dest)
    t0 = time.perf_counter()
    copy_cmd = ["cp", "-R", str(data_path), str(scratch_dir)]
    proc = subprocess.run(copy_cmd, check=True)
    t1 = time.perf_counter()
    print(f"Copied the .tar file in {t1 - t0:0.4f} seconds", flush=True)
    return str(dest)
    


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


def mark_event(step, description):
    pass
    # table = wandb.Table(columns=["epoch", "event"])
    # table.add_data(step, description)
    # log({"train_events": table}, step=step)


def save_model_checkpoint(model, optimizer, run_id: str, epoch: int, loss: float, overwrite=False, previous_run_id : str | None = None, extra_data: dict | None=None):
    """
    Saves a checkpoint to:
        $CHECKPOINT_DIR/<ModelClassName>/<run_id>/
    
    Filename format:
        checkpoint-%Y-%m-%d_%H-%M-%S

    If overwrite=True, the directory for this run_id is cleared first
    so checkpoints from the same run do not accumulate.

    Adds wandb run id to checkpoint if wandb is active.
    """
    # --- Model name from class ---
    model_name = model.__class__.__name__

    # --- Root checkpoint directory ---
    root = os.environ.get("CHECKPOINT_DIR")
    if not root:
        raise EnvironmentError("CHECKPOINT_DIR environment variable is not set.")

    # --- Per-model directory ---
    model_dir = Path(root) / model_name

    # --- Per-run directory ---
    run_dir = model_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional overwrite mode ---
    if overwrite:
        # Remove old checkpoints for this run
        for f in run_dir.iterdir():
            if f.is_file():
                f.unlink()

    # --- Check for wandb usage ---
    wandb_run_id = None
    if os.environ.get("WANDB_PROJECT", None):
        wandb_run_id = wandb.run.id

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_path = run_dir / f"checkpoint-{timestamp}"

    # --- Prepare the checkpoint dict ---
    checkpoint = {
        "model_class": model_name,
        "state_dict": model.state_dict(),
        "run_id": run_id,
        "wandb_run_id": wandb_run_id,
        "epoch": epoch,
        "loss:": loss,
        "optimizer_state": optimizer.state_dict(),
        "previous_run_id": previous_run_id
    }


    # --- Save ---
    torch.save(checkpoint, f"{ckpt_path}.pt")
    del checkpoint["state_dict"]
    del checkpoint["optimizer_state"]
    if extra_data:
        checkpoint["data"] = extra_data
    with open(f"{ckpt_path}.json", "w") as f:
        json.dump(checkpoint, f, indent=1)

    return ckpt_path

def get_latest_ckpt_path(modelname: str):
    root = os.environ["CHECKPOINT_DIR"]
    model_dir = Path(root) / modelname
    if model_dir.exists():
        # Pick lexicographically last subfolder
        subfolders = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        if subfolders:
            latest_folder = subfolders[-1]
            # Expect exactly one checkpoint file inside the folder
            ckpt_files = list([x for x in latest_folder.glob("*.pt")])
            if ckpt_files:
                return ckpt_files[0]
            
    raise ValueError("--latest specified but no existing checkpoint found")

def load_model_checkpoint(args, module: ModuleType):
    """
    Instantiate model and optionally load a checkpoint.

    Behavior:
    - Instantiate model via utils.get_nn_module_subclasses(mymodel)[args.model]()
    - If args.checkpoint is set -> load that exact path (error if missing)
    - Elif args.latest is True -> pick the lexicographically last subfolder
      inside $CHECKPOINT_DIR/<ModelClassName>/ and load the single checkpoint inside.
      If no subfolders exist -> return fresh model.
    - Raise ValueError if checkpoint model_class doesn't match args.model
    - Raise KeyError if "state_dict" is missing
    """
    # Instantiate model
    model = get_nn_module_subclasses(module)[args.model]()
    
    #Define state dict
    previous_state_dict = {"wandb_run_id": None,
                           "epoch": 0,
                           "run_id": None,
                           "optimizer_state": None}

    # No checkpoint requested -> return initialized model
    if not args.checkpoint and not args.latest:
        return model, previous_state_dict

    # Explicit checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # Latest checkpoint
    else:
        ckpt_path = get_latest_ckpt_path(args.model)

    print("loading checkpoint", ckpt_path)
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    
    # Restore wandb run id if present
    previous_state_dict["wandb_run_id"] = checkpoint.get("wandb_run_id", None)
    previous_state_dict["epoch"] = checkpoint.get("epoch", 1)
    previous_state_dict["run_id"] = checkpoint.get("run_id", None)

    # Load model weights
    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'state_dict' key.")
    model.load_state_dict(checkpoint["state_dict"])
    previous_state_dict["optimizer_state"] = checkpoint["optimizer_state"]


    return model, previous_state_dict


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