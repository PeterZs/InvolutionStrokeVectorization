import csv
import itertools
import os
import json
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Callable, Dict, Any
import cv2
import numpy as np
from PIL import Image

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def append_dict_to_csv(file_path, data_dict: dict, mainkey: dict = {}):
    """
    Appends dictionary values as a row in a CSV file.
    If the file does not exist, headers will be written first.

    :param file_path: Path to CSV file
    :param data_dict: Dictionary containing row data
    """
    file_exists = os.path.isfile(file_path)
    tmp = {**mainkey, **data_dict}
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tmp.keys())

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(tmp)

def dict_to_kv_csv(data: dict[int, int], filename: str) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for k, v in data.items():
            writer.writerow([k, v])

def kv_csv_to_dict(filename: str) -> dict[int, int]:
    result = {}
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        for k, v in reader:
            result[int(k)] = int(v)
    return result

def count_components(adj_matrix):
    # Convert to sparse matrix for efficiency (csgraph expects this)
    sparse_graph = csr_matrix(adj_matrix)
    
    # directed=False ensures we treat it as an undirected graph
    # return_labels=True returns an array where arr[i] is the label ID of node i
    n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
    
    # Count occurrences of each label to get component sizes
    _, counts = np.unique(labels, return_counts=True)
    
    return counts.tolist()

def save_webp(img_bgr, path):
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
    else:
        img_rgb = img_bgr  # grayscale

    pil_img = Image.fromarray(img_rgb)

    pil_img.save(
        path,
        format="WEBP",
        lossless=True,
    )
    

def save_polylines_to_json(lines: dict[int, np.ndarray], path: str) -> None:
    serialized_lines = {}

    for k, val in lines.items():

        # Convert numpy array → python list while preserving order
        serialized_lines[k] = val.tolist()

    # Write JSON (lists preserve order by definition)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized_lines, f, ensure_ascii=False)
        

def load_lines(data) -> list[np.ndarray]:


    lines = []

    for key_str, polyline in data.items():

        arr = np.asarray(polyline, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Polyline for key {key_str} must be of shape (N, 2), "
                f"got {arr.shape}"
            )

        lines.append(arr)

    return lines

def run_cli_tool(
    writer_callback: Callable[[str], None], 
    cli_command_builder: Callable[[str], list[str]], 
    file_ext: str = ".txt"
) -> Dict[str, Any]:
    """
    Creates a temp file, uses a callback to write data to it, runs a CLI tool, 
    and reads the resulting JSON file generated in the same directory.
    
    :param writer_callback: A function that accepts a pathlib.Path object to write data to.
    :param cli_command: The base CLI command to execute.
    :param file_ext: The extension for the temporary file (e.g., '.csv', '.bin').
    """
    
    # Ensure the extension formats cleanly (e.g., "csv" becomes ".csv")
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"
        
    # Create an isolated temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 1. Construct the input file path
        input_file_path = temp_dir_path / f"input{file_ext}"
        
        # 2. Let the callback handle writing the custom content to the file
        writer_callback(str(input_file_path))
        
        # 3. Call the CLI utility
        command = cli_command_builder(str(input_file_path))
        print(command)
        
        try:
            subprocess.run(
                command,
                check=True,          # Raises an exception if exit code != 0
                capture_output=True, # Captures stdout/stderr
                text=True            # Treats the output as strings
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CLI tool failed! Error output:\n{e.stderr}") from e
            
        # 4. Retrieve the JSON
        json_files = list(temp_dir_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError("CLI tool finished successfully, but no JSON file was found.")
            
        output_file_path = json_files[0]
        
        # Parse and return the JSON
        with open(output_file_path, "r", encoding="utf-8") as f:
            return json.load(f)