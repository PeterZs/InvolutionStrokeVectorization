import numpy as np
import cv2
from PIL import Image
import torch
import utils
import dataset


blank = torch.zeros(1, 500, 500)

img = dataset.add_noise_pattern(blank, target_mean=5.0/255.0, max_value = 20.0/255.0)
print(img.shape)

arr = utils.to_cv_img(img)
print(arr.shape)
cv2.imwrite(f"recreated.png", arr)