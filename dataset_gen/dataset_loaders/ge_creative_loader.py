import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
from dataset_loaders.loader_utils import Loader
import json
from pprint import pprint

BASE_FOLDER = os.getenv("dataset_root") + "gecreative/processed_data"
TRAIN_CUTOFF = 657

def get_svg(jsondata: dict, target_size, padding = 10):
    
    polylines= []
    for k, v in jsondata['input_parts'].items():
        for p in v: 
            points = np.array(p)
            polylines.append(
                points
            )
    
    for u in jsondata['target_part']:
        # for p in u: 
            points = np.array(u)
            polylines.append(
                points
            )
        
    mm = svgutils.get_overall_min_max(polylines)
    offset = - np.array([mm['x_min'], mm['y_min']]) + padding
    bound_x = int(mm['x_max'] - mm['x_min']) + 2*padding
    bound_y = int(mm['y_max'] - mm['y_min']) + 2*padding
    
    paths = []
    for p in polylines:
        paths.append(
            svgutils.points_to_path(p + offset)
        )
    
    return svgutils.normalized_svg_frompaths(paths, bound_x, bound_y, target_size)


class GeCreativeDataset(Loader):
    
    def sample_loader(self, split: str, target_size, index = 0):
        class_folder = [os.path.join(BASE_FOLDER, name) for name in os.listdir(BASE_FOLDER)
                        if os.path.isdir(os.path.join(BASE_FOLDER, name))]
        class_folder.sort()
        count =0
        for i in class_folder:
            tmp = os.listdir(i)
            tmp.sort()
            if split == 'train':
                tmp = tmp[:TRAIN_CUTOFF] if len(tmp) >= TRAIN_CUTOFF else tmp
            else:
                raise ValueError("not implemented")
            for file in tmp:
                if count < index:
                    count+=1
                    continue
                count+=1
                if file.endswith('.json'):
                    path = os.path.join(i, file)
                    with open(path, 'r') as f:
                        data = json.load(f)
                        yield get_svg(data, target_size)



if __name__ == "__main__":
    OUT_PATH = 'dataset_samples/ge_creative_sample.svg'

    print("hello")
    d = GeCreativeDataset()           
    for i in d.sample_loader(1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break