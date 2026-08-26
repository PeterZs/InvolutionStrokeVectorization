import os
import sys
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load configuration from config.env
script_dir = Path(__file__).parent.parent
# print(script_dir)
load_dotenv(script_dir / "config.env")

DATA_PERMANENT = os.getenv("PERMANENT_DATA_PATH")
DEST_PATH = os.getenv("DATA_ROOT")

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
        if any(p.is_file() for p in dest.rglob("*")):
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
    if platform.system() == "Darwin":   # macOS
        command = "gtar"
    else:
        command = "tar"

    tar_cmd = [command, "-xf", str(tar_file_in_scratch), "-C", str(scratch_dir)]
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
    if not DATA_PERMANENT:
        print("Error: PERMANENT_DATA_PATH not found in config.env")
        sys.exit(1)
    if not DEST_PATH:
        print("Error: DATA_ROOT not found in config.env")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Dataset initialization script")
    parser.add_argument("--datasets", nargs="*", help="List of dataset names to process")
    parser.add_argument("--mode", type=str, default="train", help="Mode subdirectory (e.g., train, test)")
    
    args, unknown = parser.parse_known_args()

    if not args.datasets:
        print("No datasets provided via --datasets.")
        return

    # Determine CPU count for parallel processing
    try:
        # Respect SLURM allocation if present
        max_workers =int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        max_workers = os.cpu_count() or 1

    print(f"Processing datasets: {args.datasets} | Workers: {max_workers}")

    # Process datasets sequentially, but process tars WITHIN a dataset in parallel
    for dataset_name in args.datasets:
        print(f"\n--- Processing Dataset Folder: {dataset_name} ---")
        
        # Source directory: PERMANENT/train/DatasetName/
        source_dataset_dir = Path(DATA_PERMANENT) / args.mode / dataset_name
        
        # Target directory: DATA_ROOT/train/DatasetName/
        # (We pass this to copy_and_unpack, which will create /images, /labels inside it)
        target_dataset_dir = Path(DEST_PATH) / args.mode / dataset_name

        if not source_dataset_dir.exists() or not source_dataset_dir.is_dir():
            print(f"Error: Directory not found: {source_dataset_dir}")
            # Depending on strictness, you might want to exit or continue
            sys.exit(1)

        # Find all tar files inside the dataset folder
        tar_files = list(source_dataset_dir.glob("*.tar"))
        
        if not tar_files:
            print(f"No .tar files found in {source_dataset_dir}")
            continue

        print(f"Found {len(tar_files)} tar files in {dataset_name}. Unpacking in parallel...")

        # Use ProcessPoolExecutor to run copy_and_unpack_tar_data in parallel
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            for tar_path in tar_files:
                # We pass the full path to the tar, and the specific dataset destination folder
                future = executor.submit(copy_and_unpack_tar_data, str(tar_path), str(target_dataset_dir))
                tasks.append(future)

            # Wait for completion and handle potential errors
            for future in as_completed(tasks):
                try:
                    result_path = future.result()
                    # print(f"Finished: {result_path}") 
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    sys.exit(1)

    print("\nAll processing complete.")
    print(f"Data Root is: {Path(DEST_PATH) / args.mode}")

if __name__ == "__main__":
    main()