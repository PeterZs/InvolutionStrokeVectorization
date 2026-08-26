import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
import random
import math
from dataset_loaders.loader_utils import Loader

BASE_FOLDER = os.getenv("dataset_root") + "quickdraw"
#There are 345 classes and each class has 70k samples. 
# We want ~20k samples total, so we take this many samples in each class
TRAIN_CUTOFF = 58
# 1725 samples
TEST_CUTOFF = 5
#drawing border padding.
MIN_PADDING = 0

def get_svg(data, padding: int = 10, target_size : int = 100) -> str:
    paths = []
    abs_x = 0.0
    abs_y = 0.0
    curr = [[abs_x, abs_y]]
    abspoints = np.cumsum(data[:, :2], axis=0)
    min_x, max_x = np.min(abspoints[:, 0]), np.max(abspoints[:, 0])
    min_y, max_y = np.min(abspoints[:, 1]), np.max(abspoints[:, 1])
    bound_x = int(max_x - min_x) + 2*padding
    bound_y = int(max_y - min_y) + 2*padding
    offset = - np.array([min_x, min_y]) + padding
    for p, lift_pen in zip(abspoints, data[:, 2]):
        curr.append(p)
        if lift_pen:
            paths.append(
                svgutils.points_to_path(np.array(curr)+offset)
                )
            curr = []
    return svgutils.normalized_svg_frompaths(paths, bound_x, bound_y, target_size)
    

class QuickDrawDataset(Loader):
    def sample_loader(self, split: str, target_size, index: int = 0):
        npz_files = [name for name in os.listdir(BASE_FOLDER)
             if not os.path.isdir(os.path.join(BASE_FOLDER, name)) and name.endswith('.npz')]
        npz_files.sort()

        # Dynamic cutoff based on the split to ensure correct file seeking
        if split == 'train':
            chunk_size = TRAIN_CUTOFF
            left, right = 0, TRAIN_CUTOFF
        elif split == 'test':
            chunk_size = TEST_CUTOFF
            left, right = TRAIN_CUTOFF, TRAIN_CUTOFF + TEST_CUTOFF
        else:
            raise ValueError("Only 'train' and 'test' splits are implemented")

        # Calculate starting file index and global count based on the specific chunk size
        idx = index // chunk_size
        count = idx * chunk_size

        for npz in npz_files[idx:]:
            data = np.load(os.path.join(BASE_FOLDER, npz), encoding='latin1', allow_pickle=True)['train']
            for i in data[left:right]:
                if count < index:
                    count += 1
                    continue
                count += 1
                yield get_svg(i, padding=MIN_PADDING + 2*(count%5), target_size=target_size)



if __name__ == "__main__":
    OUT_PATH = 'dataset_samples/quickdraw_sample.svg'


    d = QuickDrawDataset()           
    for i in d.sample_loader("train", 1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break
        

