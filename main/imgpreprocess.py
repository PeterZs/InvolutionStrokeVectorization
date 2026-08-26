import cv2
import numpy as np
from PIL import Image
import utils.utils as utils

def increase_contrast(image, alpha=1.5, beta=0):
    """
    alpha > 1 increases contrast
    beta shifts brightness
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def boost_brightness(image, factor=1.5):
    """
    Boost the brightness of a grayscale image.

    Parameters:
        image (numpy.ndarray): Grayscale uint8 image.
        factor (float): Brightness multiplier. >1 increases brightness.

    Returns:
        numpy.ndarray: Brightness-boosted image.
    """
    # Convert to float to prevent clipping during multiplication
    img_float = image.astype(np.float32)
    
    # Multiply pixel values by factor
    brightened = img_float * factor
    
    # Clip to 0-255 and convert back to uint8
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    return brightened


def bright_foreground(
    img,
    kernel_size=51,
    threshold=None
):
    """
    Enhance bright/white foreground while suppressing uneven background.

    Parameters
    ----------
    image : numpy array or path
        Grayscale image
    kernel_size : int
        Size of structuring element (should be larger than foreground features)
    threshold : int or None
        If set, returns binary mask

    Returns
    -------
    numpy.ndarray
        Enhanced image or binary mask
    """


    # Structuring element (controls background scale removal)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # White top-hat: original - background estimate
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Optional threshold
    # if threshold is not None:
    #     _, tophat = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)

    return tophat

def background_subtract(image, blur_size=51):

    bg = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    result = cv2.subtract(image, bg)

    return result



def preprocess(image: Image.Image, invert = True, skip_norm = False):
    assert image.mode == "L"
    img = utils.pil_to_cv2(image)
    if invert:
        img = 255-img
    if skip_norm:
        return Image.fromarray(img)
    result = bright_foreground(img, kernel_size=21)
    # result = boost_brightness(result, factor=2)
    result = background_subtract(result)
    return Image.fromarray(result)
    

if __name__ == "__main__":
    img = cv2.imread("debug_files/man_shadow.png", cv2.IMREAD_GRAYSCALE)
    img = 255-img

    result = bright_foreground(img, kernel_size=21)
    # result = boost_brightness(result, factor=2)
    result = background_subtract(result)
    cv2.imwrite("debug_files/output.png", result)