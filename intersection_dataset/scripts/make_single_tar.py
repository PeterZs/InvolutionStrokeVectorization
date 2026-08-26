#!/usr/bin/env python3
import os
import subprocess
import argparse
import tempfile
import platform
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tar creation with relative directory stripping")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--tar_dir", type=str, required=True, help="Output tar file path")
    parser.add_argument("--ext", type=str, default="", help="File extension to filter (e.g. .txt)")
    parser.add_argument("--level", type=int, default=1, help="Number of directory levels to strip inside base_dir")
    args = parser.parse_args()

    extension = args.ext.lower()
    base_path = Path(args.base_dir).resolve()

    if not base_path.exists():
        print(f"Error: Base directory '{base_path}' does not exist.")
        return

    # Collect files
    print(f"Scanning for '{extension if extension else '*'}' files in {base_path}...")
    
    files_relative = []
    
    # 1. Get files relative to base_dir
    for p in base_path.rglob(f"*{extension}"):
        if p.is_file():
            # relative_to ensures we get "a/b/file.txt" instead of full path
            rel_path = p.relative_to(base_path).as_posix()
            files_relative.append(rel_path)

    print(f"Found {len(files_relative)} files")
    
    if not files_relative:
        print("No files found. Exiting.")
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

        # 2. Construct Regex by repeating the pattern explicitly
        # Pattern unit: [^/]*/  (Any characters that aren't a slash, followed by a slash)
        # If level=2, this becomes: s|^[^/]*/[^/]*/||
        regex_pattern = "[^/]*/" * args.level
        transform_regex = f"s|^{regex_pattern}||"

        cmd = [
            command, 
            '-cvf', args.tar_dir,
            '-C', str(base_path),       # Change directory to base_dir
            '--transform', transform_regex,
            '--null', '-T', filelist_path,
        ]

        print(f"Base Directory: {base_path}")
        print(f"Transform Regex: '{transform_regex}'")
        
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
        print(f"\nCreated {args.tar_dir} successfully!")
    finally:
        if os.path.exists(filelist_path):
            os.unlink(filelist_path)

if __name__ == "__main__":
    main()