from PIL import Image


class ViolationDetector:
    
    # Dummy detector: one box in the center marked as 'nudity_explicit'.
    
    def __init__(self):
        pass

    def predict(self, image: Image.Image):
        w, h = image.size
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)
        return [
            {
                "label": "nudity_explicit",
                "score": 0.95,
                "bbox": [x1, y1, x2, y2],
            }
        ]
