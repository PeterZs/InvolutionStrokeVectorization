import os
import torch
from pathlib import Path
import cv2
import numpy as np
import sys
import os
import itertools
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import utils
import dataset


utils.load_env()
utils.set_seed(42)



dataset_path = Path(os.environ["MOCK_DATA"]) / "train"/ "TUBerlinDatasetSmall"
d =  {"dummy": str(dataset_path)}
B = 2
OUT = "tmp"
SIZE = 592
train_data, _ = dataset.get_dataloaders(d,image_size=SIZE, batch_size=B, val_split=0, num_workers=0)

for idx, batch in enumerate(itertools.islice(train_data, 12)):
    print(idx)
    input_img, output_img, mask, base_name = batch
    for k in range(B):
       # print("test img", batch["img"].dtype)
        #print("test seg", batch["seg"].dtype)
        print(input_img.shape)
        img = utils.to_cv_img(input_img[k])
        print(img.shape)
        cv2.imwrite(f"{OUT}/{base_name[k]}.png", img)
