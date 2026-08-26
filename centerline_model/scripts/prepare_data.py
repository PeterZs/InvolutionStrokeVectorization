import os
import sys
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load configuration from config.env
script_dir = Path(__file__).parent.parent
load_dotenv(script_dir / "config.env")

DATA_ROOT = os.getenv("DATA_PATH")
SCRATCH_PATH = os.getenv("SCRATCH_DIR")

def copy_and_unpack_tar_data(data_path_str, scratch_dir_str, ignore_existing=False):
    """
    Copy tar data to scratch directory and extract it.
    """
    data_path = Path(data_path_str)
    scratch_dir = Path(scratch_dir_str)
    
    dest = scratch_dir / data_path.stem
    
    # Check if already extracted
    if dest.exists() and not ignore_existing:
        # Check recursively for webp files (using rglob) to support sharded structures
        # Note: 'any' is lazy, so this is efficient enough
        if any(dest.rglob("*.webp")):
            print(f"Data for {data_path.stem} already copied and extracted at {dest}")
            return str(dest)
        else:
            print(f"Warning: {dest} exists but seems empty. Re-extracting.")

    os.makedirs(scratch_dir, exist_ok=True)

    # Step 1: Copy tar file to scratch
    t0 = time.perf_counter()
    print(f"Copying {data_path} to {scratch_dir}...")
    
    # Using cp -R to be safe, though it's a file
    copy_cmd = ["cp", str(data_path), str(scratch_dir)]
    try:
        subprocess.run(copy_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        sys.exit(1)
        
    t1 = time.perf_counter()
    print(f"Copied the .tar file in {t1 - t0:0.4f} seconds", flush=True)

    # Step 2: Extract tar file
    tar_file_in_scratch = scratch_dir / data_path.name
    
    os.makedirs(dest, exist_ok=True)
    
    tar_cmd = ["tar", "-xf", str(tar_file_in_scratch), "-C", str(dest)]
    print(" ".join(tar_cmd))
    try:
        subprocess.run(tar_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting tar: {e}")
        if tar_file_in_scratch.exists():
            os.remove(tar_file_in_scratch)
        sys.exit(1)

    t2 = time.perf_counter()
    
    if tar_file_in_scratch.exists():
        os.remove(tar_file_in_scratch)
        
    print(f"Extracted the .tar file in {t2 - t1:0.4f} seconds", flush=True)
    print(f"successfully created '{str(dest)}'")

    return str(dest)

def main():
    if not DATA_ROOT:
        print("Error: DATA_PATH not found in config.env")
        sys.exit(1)
    if not SCRATCH_PATH:
        print("Error: SCRATCH_DIR not found in config.env")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Dataset initialization script")
    parser.add_argument("--datasets", nargs="*", help="List of dataset names to process")
    parser.add_argument("--mode", type=str, default="train", help="Mode subdirectory (e.g., train, test)")
    
    # Ignore unknown arguments (e.g., training flags)
    args, unknown = parser.parse_known_args()

    if not args.datasets:
        print("No datasets provided via --datasets.")
        return

    print(f"Processing datasets: {args.datasets}")

    for dataset_name in args.datasets:
        print(f"\n--- Processing: {dataset_name} ---")
        
        # Paths
        source_tar = os.path.join(DATA_ROOT, args.mode, f"{dataset_name}.tar")
        source_csv = os.path.join(DATA_ROOT, args.mode, f"{dataset_name}.csv")
        scratch_mode_dir = os.path.join(SCRATCH_PATH, args.mode)

        if not os.path.exists(source_tar):
            raise FileNotFoundError(f"Error: Tar file not found: {source_tar}")

        # 1. Copy and Unpack
        dest_dir = copy_and_unpack_tar_data(source_tar, scratch_mode_dir)

        # 2. Copy the CSV file
        dest_csv_path = dest_dir+".csv"
        
        if os.path.exists(source_csv):
            print(f"Copying CSV to '{dest_csv_path}'...")
            shutil.copy(source_csv, dest_csv_path)
        else:
            raise FileNotFoundError(f"Warning: Source CSV not found at {source_csv}")

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()