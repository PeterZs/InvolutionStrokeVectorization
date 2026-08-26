import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
from dataset_loaders.loader_utils import Loader
BASE_FOLDER = os.getenv("dataset_root") + "TU berlin Sketch Dataset/svg"
TRAIN_CUTOFF = 19000

class TUBerlinDataset(Loader):
    def sample_loader(self, split, target_size = 1000, index = 0):
        
        with open(BASE_FOLDER+"/filelist.txt", 'r') as f:
            files = f.readlines()
        files.sort()
        if split == 'train':
            start, end = index, TRAIN_CUTOFF
        else:
            start, end = TRAIN_CUTOFF+index, len(files)
        # folders = [name for name in os.listdir(BASE_FOLDER)
        #         if os.path.isdir(os.path.join(BASE_FOLDER, name))]
        for svg in files[start:end]:
            with open(BASE_FOLDER+"/" + svg.strip(), 'r') as f:
                svg = f.read()
                yield svgutils.normalize_svg(svg, target_size)
            

if __name__ == "__main__":
    OUT_PATH = 'dataset_samples/tuberlin_sample.svg'
    d = TUBerlinDataset()           
    for i in d.sample_loader('train', target_size=1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break
        