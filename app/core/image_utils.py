from PIL import Image
import numpy as np
import cv2


def blur_with_mask(pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
    
    img = np.array(pil_image) 

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)    # if grayscale, make 3-channel

    h, w = img.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(f"Mask shape {mask.shape} != image shape {(h, w)}")

    blurred = cv2.GaussianBlur(img, (31, 31), 0)

    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    out = img.copy()
    out[mask_3ch] = blurred[mask_3ch]

    return Image.fromarray(out)