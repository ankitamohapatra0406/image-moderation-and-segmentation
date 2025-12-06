from PIL import Image
import numpy as np


class ViolationSegmenter:
    
    # Dummy segmenter: mask is just the bounding box rectangle.
  
    def __init__(self):
        pass

    def segment(self, image: Image.Image, bbox):
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        mask[y1:y2, x1:x2] = 1
        return mask
