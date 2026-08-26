import torch
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import math
from PIL import Image
import utils.utils as utils



def load_model() -> torch.ScriptModule:
    model_path = os.environ["CENTERLINE_MODEL"]
    # This loads the logic AND the weights
    model = torch.jit.load(model_path)
    utils.get_param_count(model)
    return model



def prepare_image(image):
    c = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                        transforms.ToTensor()])
    
    return c(image)

def next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(n)) if n > 0 else 1

def pad_to_power_of_2(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = tensor.shape
    new_h = next_power_of_2(h)
    new_w = next_power_of_2(w)
    pad_bottom = new_h - h
    pad_right = new_w - w
    padded = torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
    return padded, (h, w)

def predict_centerline(model: torch.ScriptModule, image: Image.Image, invert=False) -> Image.Image:
    image.verify()
    img = prepare_image(image).unsqueeze(0)
    if invert:
        img = 1 - img
    img, orig_size = pad_to_power_of_2(img)
    if torch.backends.mps.is_available():
        d = torch.device("mps")
        img = img.to(d)
        model = model.to(d)
    with torch.no_grad():
        prediction = model(img)
    oh, ow = orig_size
    prediction = prediction[:, :, :oh, :ow]
    return F.to_pil_image(prediction[0])


if __name__ == "__main__":
    model = load_model()
    image_path = "debug_files/man.png"
    image =  Image.open(image_path).convert("RGB")
    out = predict_centerline(model, image, invert=True)
    out.save("output.png")