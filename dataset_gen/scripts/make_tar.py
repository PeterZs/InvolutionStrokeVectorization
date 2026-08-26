#!/usr/bin/env python3
import os
import subprocess
import argparse
import tempfile
import platform
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tar creation using GNU tar")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--tar_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".webp")
    args = parser.parse_args()

    extension = args.ext.lower()
    
    # Collect files
    print(f"Scanning for '{extension}' files...")
    files = [str(p) for p in Path(args.base_dir).rglob(f"*{extension}")]
    print(f"Found {len(files)} files")

    # Write file list to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\0'.join(files))
        filelist_path = f.name

    try:
        if platform.system() == "Darwin":   # macOS
            command = "gtar"
        else:
            command = "tar"

        # Use GNU tar with --files-from (much faster than Python tarfile)
        cmd = [
            command, '-cvf', args.tar_dir,
            '--transform', 's|.*/||',  # Strip directory paths
            '--null', '-T', filelist_path,
            '--checkpoint=100',
            '--checkpoint-action=echo="%u blocks"'
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
        print(f"\nCreated {args.tar_dir} successfully!")
    finally:
        os.unlink(filelist_path)

if __name__ == "__main__":
    main()