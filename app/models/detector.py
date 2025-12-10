# from PIL import Image


# class ViolationDetector:
    
#     # Dummy detector: one box in the center marked as 'nudity_explicit'.
    
#     def __init__(self):
#         pass

#     def predict(self, image: Image.Image):
#         w, h = image.size
#         x1, y1 = int(w * 0.3), int(h * 0.3)
#         x2, y2 = int(w * 0.7), int(h * 0.7)
#         return [
#             {
#                 "label": "nudity_explicit",
#                 "score": 0.95,
#                 "bbox": [x1, y1, x2, y2],
#             }
#         ]

# app/models/detector.py

# app/models/detector.py
import os
from typing import List, Dict, Union
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class ViolationDetector:
    """
    YOLOv8 wrapper that loads weights from models/weights/detector_best.pt
    and exposes a .predict(image) method.

    Accepts image as:
      - PIL.Image.Image
      - file path (str)
      - numpy array (H,W,C) uint8

    Returns list of dicts:
      [{ "label": "knife", "score": 0.92, "bbox": [x1, y1, x2, y2] }, ...]
    """

    def __init__(self, weights_filename: str = "detector_best.pt", conf: float = 0.25):
        if YOLO is None:
            raise RuntimeError(
                "ultralytics package not found. Install it with `pip install ultralytics`."
            )

        base_dir = os.path.dirname(__file__)
        self.weights_path = os.path.join(base_dir, "weights", weights_filename)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found at {self.weights_path}")

        self._model = None
        self._conf = float(conf)
        self._names = None

    def _load_model(self):
        if self._model is None:
            # instantiate YOLO model (this loads weights)
            self._model = YOLO(self.weights_path)
            # Try to grab names (class id -> label)
            self._names = getattr(self._model, "names", None)
            if self._names is None:
                # fallback to wrapped model attribute (API differences)
                try:
                    self._names = getattr(self._model.model, "names", None)
                except Exception:
                    self._names = None

    def _ensure_image(self, image: Union[Image.Image, str, np.ndarray]) -> Union[Image.Image, str, np.ndarray]:
        # if path string, return as-is (ultralytics accepts file paths)
        if isinstance(image, str):
            return image
        # numpy array - pass-through
        if isinstance(image, np.ndarray):
            return image
        # PIL - ensure RGB
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        raise TypeError("Unsupported image type. Provide PIL.Image, image path (str) or numpy array.")

    def predict(self, image: Union[Image.Image, str, np.ndarray], conf: float = None) -> List[Dict]:
        """
        Run detection and return list of dicts with keys: label, score, bbox (x1,y1,x2,y2)
        """
        self._load_model()
        conf = self._conf if conf is None else float(conf)

        img = self._ensure_image(image)

        # run prediction; ultralytics returns a list of Results for each input (we pass one)
        results = self._model.predict(source=img, conf=conf, verbose=False)

        if not results:
            return []

        r = results[0]  # single image result
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return []

        # Convert tensors to numpy when available
        # boxes.xyxy, boxes.conf, boxes.cls
        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy = np.array(boxes.xyxy)

        try:
            scores = boxes.conf.cpu().numpy()
        except Exception:
            scores = np.array(boxes.conf)

        try:
            cls_inds = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cls_inds = np.array(boxes.cls).astype(int)

        out = []
        for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, cls_inds):
            if self._names:
                label = self._names[int(cls)] if int(cls) in self._names else str(int(cls))
            else:
                label = str(int(cls))

            out.append({
                "label": label,
                "score": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return out

