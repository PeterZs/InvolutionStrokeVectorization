import argparse
import sys
from pathlib import Path
import polars as pl
import polars.selectors as cs

def main():
    # 1. Set up command-line arguments to accept the folder path
    parser = argparse.ArgumentParser(description="Sum all columns across stats.csv files in subfolders.")
    parser.add_argument("folder", type=str, help="Path to the parent folder")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: The directory '{args.folder}' does not exist.")
        sys.exit(1)

    # 2. Find stats.csv strictly in immediate subfolders (not fully recursively)
    # The pattern "*/stats.csv" looks exactly one level deep.
    file_paths = list(folder_path.glob("*/stats.csv"))

    if not file_paths:
        print(f"No 'stats.csv' files found in immediate subfolders of '{args.folder}'.")
        sys.exit(0)

    print(f"Found {len(file_paths)} 'stats.csv' files. Processing...")

    # Convert Path objects to strings for Polars
    file_paths_str = [str(p) for p in file_paths]

    try:
        # 3. Use Polars Lazy API to prevent loading all 100k rows into memory at once.
        # pl.scan_csv builds a query plan without executing it yet.
        lazy_df = pl.scan_csv(file_paths_str)
        
        # 4. Sum the columns. 
        query = lazy_df.select(cs.numeric().sum())
        
        # 5. Execute the query
        result_df = query.collect(engine="streaming")
        
        # 6. Print the results nicely
        print("\n--- Column Sums ---")
        
        # Convert the single-row DataFrame to a dictionary for clean printing
        sums_dict = result_df.to_dicts()[0]
        for col_name, total in sums_dict.items():
            # Format numbers to be easily readable
            print(f"{col_name:<20}: {total}")
            
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        print("Note: Ensure all stats.csv files have the same column names and data types.")

if __name__ == "__main__":
    main()