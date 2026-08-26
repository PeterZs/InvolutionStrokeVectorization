#!/usr/bin/env python3
import os
import subprocess
import argparse
import tempfile
import platform
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Shard-aware Tar creation")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory containing shards")
    parser.add_argument("--tar_dir", type=str, required=True, help="Output directory where .tar files will be saved")
    parser.add_argument("--name", type=str, required=True, help="The specific subfolder name to target (e.g. 'a' for a.tar)")
    parser.add_argument("--ext", type=str, default="", help="File extension to filter (e.g. .txt)")
    args = parser.parse_args()

    extension = args.ext.lower()
    base_path = Path(args.base_dir).resolve()
    output_dir = Path(args.tar_dir).resolve()
    target_name = args.name

    if not base_path.exists():
        print(f"Error: Base directory '{base_path}' does not exist.")
        return
    os.makedirs(args.tar_dir, exist_ok=True)
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output tar filename: output_dir/name.tar
    output_tar_file = output_dir / f"{target_name}.tar"

    print(f"Scanning for '{target_name}' folders inside shards of {base_path}...")
    
    files_relative = []
    
    # Logic: 
    # 1. Iterate over immediate children of base_dir (the "shards")
    # 2. Check if {shard}/{target_name} exists
    # 3. Collect files recursively from there
    # 4. Store path relative to base_dir (e.g. "shard1/a/b.txt")
    
    # Get all immediate subdirectories (shards)
    shards = [x for x in base_path.iterdir() if x.is_dir()]
    
    for shard in shards:
        target_subfolder = shard / target_name
        
        if target_subfolder.exists() and target_subfolder.is_dir():
            # Scan inside shard/name
            for p in target_subfolder.rglob(f"*{extension}"):
                if p.is_file():
                    # relative_to base_path ensures we get "shard1/a/b.txt"
                    rel_path = p.relative_to(base_path).as_posix()
                    files_relative.append(rel_path)

    print(f"Found {len(files_relative)} files for target '{target_name}'")
    
    if not files_relative:
        print(f"No files found for category '{target_name}'. Skipping.")
        return

    # Write file list to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\0'.join(files_relative))
        filelist_path = f.name

    try:
        if platform.system() == "Darwin":   # macOS
            command = "gtar"
        else:
            command = "tar"

        # Regex to strip exactly 1 level (the shard name).
        # Example: "shard1/a/b.txt" -> "a/b.txt"
        # Regex: s|^[^/]*/||
        regex_pattern = "[^/]*/" # Matches "text/"
        transform_regex = f"s|^{regex_pattern}||"

        cmd = [
            command, 
            '-cvf', str(output_tar_file),
            '-C', str(base_path),       # Change directory to base_dir
            '--transform', transform_regex,
            '--null', '-T', filelist_path,
        ]

        print(f"Target Tar: {output_tar_file}")
        print(f"Transform Regex: '{transform_regex}'")
        
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
        print(f"\nCreated {output_tar_file.name} successfully!")
    finally:
        if os.path.exists(filelist_path):
            os.unlink(filelist_path)

if __name__ == "__main__":
    main()