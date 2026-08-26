import os
import torch
from pathlib import Path
import cv2
import numpy as np
import sys
import os
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import utils
import metrics_and_losses
import dataset
import itertools


utils.load_env()

def colorize_index_image(index_img: torch.Tensor, color_generator):
    # Ensure tensor is on CPU
    index_img = index_img.cpu()

    # Get unique region indices
    unique_indices = torch.unique(index_img)

    # Create output image (OpenCV format)
    h, w = index_img.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to numpy once for masking
    index_np = index_img.numpy()

    # Assign a new color per unique index
    for idx in unique_indices:
        if idx == 0:
            continue
        color = next(color_generator)  # Already (B, G, R)
        mask = index_np == idx.item()
        output[mask] = color

    return output

dataset_path = Path(os.environ["DATA_ROOT"]) / "train"/"Dataset2"
d =  {"SampleDataset": str(dataset_path)}
B = 2
OUT = "debug_files"
train_data, _ = dataset.get_dataloaders(d,batch_size=B, val_split=0, num_workers=0)
colors = utils.color_gen(rgb=False)

for idx, batch in enumerate(itertools.islice(train_data, 1)):
    print(idx, batch["basename"])
    for k in range(B):
       # print("test img", batch["img"].dtype)
        #print("test seg", batch["seg"].dtype)
        img = utils.to_cv_img(batch["img"][k])
        print(img.shape)
        imgcolor = colorize_index_image(batch["seg"][k], colors)
        cv2.imwrite(f"{OUT}/{batch['basename'][k]}.png", img)
        cv2.imwrite(f"{OUT}/{batch['basename'][k]}_indexes.png", imgcolor)
