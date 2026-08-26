import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
from dataset_loaders.loader_utils import Loader
import json
from pprint import pprint
BASE_FOLDER = os.getenv("dataset_root") + "fscoco/vector_sketches"


def get_svg(data, target_size : int, padding: int = 10) -> str:
    paths = []
    abs_x = 0.0
    abs_y = 0.0
    curr = [[abs_x, abs_y]]
    abspoints = data[:, :2]
    min_x, max_x = np.min(abspoints[:, 0]), np.max(abspoints[:, 0])
    min_y, max_y = np.min(abspoints[:, 1]), np.max(abspoints[:, 1])
    bound_x = int(max_x - min_x) + 2*padding
    bound_y = int(max_y - min_y) + 2*padding
    offset = - np.array([min_x, min_y]) + padding
    for p, lift_pen in zip(abspoints, data[:, 2]):
        curr.append(p)
        if lift_pen:
            if len(curr) == 1:
                continue
            paths.append(
                svgutils.points_to_path(np.array(curr)+offset)
                )
            curr = []
            
    return svgutils.normalized_svg_frompaths(paths, bound_x, bound_y, target_size)


class FSCocoDataset(Loader):
    
    def sample_loader(self, split: str, target_size: int, index = 0):
        class_folder = [os.path.join(BASE_FOLDER, name) for name in os.listdir(BASE_FOLDER)
                        if os.path.isdir(os.path.join(BASE_FOLDER, name))]
        class_folder.sort()
        for i in class_folder:
            for file in sorted(os.listdir(i)):
                if file.endswith('.npy'):
                    print(file)
                    path = os.path.join(i, file)
                    data = np.load(path)
                    yield get_svg(data, target_size)
                        



if __name__ == "__main__":
    OUT_PATH = 'dataset_samples/fscoco_sample.svg'

    print("hello")
    d = FSCocoDataset()           
    for i in d.sample_loader(1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break