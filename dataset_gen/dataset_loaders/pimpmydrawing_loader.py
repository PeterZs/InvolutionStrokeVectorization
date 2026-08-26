import subprocess
import tempfile
import os
import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
from dataset_loaders.loader_utils import Loader

BASE_FOLDER = os.getenv("dataset_root") + "pimpmydrawings"

INKSCAPE_PATH = '/Applications/Inkscape.app/Contents/MacOS/inkscape'

def convert_to_svg(input_path, inkscape_path):
    # Create a temporary file to store the SVG output
    temp_file = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # Run Inkscape subprocess to convert input to plain SVG
        subprocess.run([
            inkscape_path,
            input_path,
            # '--actions=select-all;object-release-clip;',
            "--export-plain-svg",
            "--export-type=svg",
            f"--export-filename={temp_file_path}"
        ], check=True)
        

        # Read the SVG content as a string
        with open(temp_file_path, "r", encoding="utf-8") as f:
            svg_string = f.read()

    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    tmp = svgutils.remove_clip_paths(svg_string)
    return  svgutils.remove_duplicated(tmp)

class PimpMyDrawingDataset(Loader):
    def sample_loader(self, split, target_size=1000, index=0):
        if split == 'train':
            class_folder = ['people', 'people-top', 'tree', 'cars']
        else:
            raise ValueError("specify classes for test set")

        count = 0 

        for i in class_folder:
            path = os.path.join(BASE_FOLDER, i)
            files = sorted([f for f in os.listdir(path) if f.endswith('.ai')])
            for file in files:
                if count < index:
                    count += 1
                    continue

                filepath = os.path.join(path, file)
                svg = convert_to_svg(filepath, INKSCAPE_PATH)
                yield svgutils.normalize_svg(svg, target_size)
                count += 1



if __name__ == "__main__":
    
    OUT_PATH = 'dataset_samples/pimpmydrawing_sample.svg'
    d = PimpMyDrawingDataset()         
    for i in d.sample_loader('train', target_size =1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break
    
    