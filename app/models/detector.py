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

import os
from typing import List, Dict, Any, Optional

from PIL import Image

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None  
    _HAS_ULTRALYTICS = False


DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "weights", "detector_best.pt"
)


class ViolationDetector:
    """
    Loads a YOLO model (ultralytics) and exposes .predict(pil_image) -> List[dict]
    Each dict: {"label": str, "score": float, "bbox": [x1,y1,x2,y2]}
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS
        self.weights_path = weights_path
        self.device = device 

        if _HAS_ULTRALYTICS:
            if os.path.exists(self.weights_path):
                # YOLO will auto-download a model if a known tag is passed,
                # but here we're pointing at local weights (best.pt)
                try:
                    if self.device:
                        self.model = YOLO(self.weights_path, device=self.device)  
                    else:
                        self.model = YOLO(self.weights_path) 
                    self.names = getattr(self.model, "names", {})
                    self.ready = True
                except Exception:
                    self.model = None
                    self.names = {}
                    self.ready = False
            else:
                self.model = None
                self.names = {}
                self.ready = False
        else:
            self.model = None
            self.names = {}
            self.ready = False

    def predict(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run detection on a PIL image and return detections in the format:
        [{ "label": "cigarette", "score": 0.92, "bbox": [x1, y1, x2, y2] }, ...]
        """
        if not self.ready or self.model is None:
            return []

        # ultralytics models accept PIL images directly
        try:
            results = self.model(pil_image)  # run inference
        except Exception:
            return []

        out = []
        # results is an ultralytics.Results object (sequence). Usually first element is for the input image.
        # Iterate found boxes in results[0].boxes
        try:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                return out

            # each box has .xyxy, .conf, .cls attributes
            for b in boxes:
                # xyxy may be a tensor with shape (4,) or (1,4) depending on version; handle robustly
                try:
                    xy = b.xyxy[0].tolist()
                except Exception:
                    # fallback: convert whole xyxy to list and use first entry
                    try:
                        xy = list(map(float, b.xyxy.tolist()[0]))
                    except Exception:
                        continue
                # convert to floats
                x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])

                # confidence and class
                try:
                    conf = float(b.conf[0]) if hasattr(b, "conf") else float(b.conf)
                except Exception:
                    # try alternative attribute
                    conf = float(getattr(b, "confidence", 0.0) or 0.0)

                try:
                    cls_idx = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
                except Exception:
                    cls_idx = None

                label = str(self.names.get(cls_idx, cls_idx)) if cls_idx is not None else "unknown"

                out.append({"label": label, "score": conf, "bbox": [x1, y1, x2, y2]})
        except Exception:
            return out

        return out


# convenience factory so other code can do `from app.models.detector import get_detector`
def get_detector(weights_path: Optional[str] = None, device: Optional[str] = None) -> ViolationDetector:
    return ViolationDetector(weights_path=weights_path, device=device)
