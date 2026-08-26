import os
import sys
import numpy as np
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)
import svgutils
from dataset_loaders.loader_utils import Loader
BASE_FOLDER = os.getenv("dataset_root") + "instancesegmentation"


class InstanceSegmentationDataset(Loader):
    def sample_loader(self, split, target_size=1000, index = 0):
        
        class_folder = [os.path.join(BASE_FOLDER, f'sketchagent_{split}'), os.path.join(BASE_FOLDER, f'clipasso-base_{split}')]
        count = 0
        for i in class_folder:
            for file in sorted(os.listdir(os.path.join(i, 'DRAWING_GT_SVG'))):
                if not file.endswith('.svg'): continue
                if count < index:
                    count+=1
                    continue
                count+=1
                with open(os.path.join(i, 'DRAWING_GT_SVG', file), 'r') as f:
                        svg = f.read()
                        yield svgutils.normalize_svg(svg, target_size)
            

if __name__ == "__main__":
    
    OUT_PATH = 'dataset_samples/instancesegmentation_sample.svg'
    d = InstanceSegmentationDataset()           
    for i in d.sample_loader('train', target_size =1000):
        with open(OUT_PATH, 'w') as f:
            f.write(i)
        break
        