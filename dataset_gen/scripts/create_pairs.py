import os
import argparse
import csv
from pathlib import Path

def get_pairs_in_directory(directory, relative_path_prefix):
    """
    Identifies input/output pairs in a specific directory based on the naming convention:
    Input:  prefix_part1_part2.webp
    Output: prefix_part1_part2_output.webp
    """
    pairs = []
    output_map = {}  # prefix -> filename

    # List webp files in the current directory
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".webp")]
    except OSError:
        return []

    # First pass: collect output files
    for filename in files:
        name_no_ext = filename.rsplit(".", 1)[0]
        if name_no_ext.endswith("_output"):
            # Remove '_output' and any trailing underscores
            prefix = name_no_ext[:-7].rstrip("_")
            output_map[prefix] = filename

    # Second pass: pair inputs with outputs
    for filename in files:
        name_no_ext = filename.rsplit(".", 1)[0]

        if name_no_ext.endswith("_output"):
            continue  # skip outputs

        # Validate input filename format (must have at least 3 parts separated by underscores)
        parts = name_no_ext.split("_", 2)
        if len(parts) < 3:
            continue

        prefix = parts[0] + "_" + parts[1]

        if prefix in output_map:
            # We found a match
            input_rel = filename
            output_rel = output_map[prefix]
            
            pairs.append((input_rel, output_rel))

    return pairs

def main():
    parser = argparse.ArgumentParser(description="Generate image pairs CSV recursively.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--dest", type=str, help="Destination path for the output .csv file. Defaults to {root}.csv")
    
    args = parser.parse_args()
    
    # Logic to set default destination if not provided
    if args.dest is None:
        # normpath ensures "my_data/" becomes "my_data.csv" instead of "my_data/.csv"
        args.dest = f"{os.path.normpath(args.root)}.csv"
    
    root_path = args.root
    total_pairs = 0
    
    print(f"Scanning {root_path}...")

    # Write incrementally to handle large datasets
    with open(args.dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Recursive walk to handle shards/subdirectories
        # This respects the 'top-level folder' hint by processing the tree structure naturally
        for current_dir, dirs, files in os.walk(root_path):
            
            # Calculate path relative to the root (e.g., 'shard1' or 'shard1/sub')
            rel_path = os.path.relpath(current_dir, root_path)
            if rel_path == ".":
                rel_path = ""
            
            # Optimization: Only process directory if it contains .webp files
            if any(file.endswith(".webp") for file in files):
                print(f"Getting pairs in {current_dir}")
                dir_pairs = get_pairs_in_directory(current_dir, rel_path)
                
                if dir_pairs:
                    writer.writerows(dir_pairs)
                    total_pairs += len(dir_pairs)

    print(f"Completed. Found {total_pairs} pairs.")
    print(f"CSV saved to: {args.dest}")

if __name__ == "__main__":
    main()