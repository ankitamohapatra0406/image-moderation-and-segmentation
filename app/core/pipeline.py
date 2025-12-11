import os
import uuid
from typing import Dict, Any, List

import numpy as np
from PIL import Image

from app.models import NSFWClassifier, ViolationDetector, ViolationSegmenter
from app.core.image_utils import blur_with_mask


class ModerationPipeline:
    """
    1. Classify → possible violation?
    2. Detect → violating boxes
    3. Segment → precise masks
    4. Blur → only masked regions
    """

    def __init__(self, output_dir: str = "safe_images"):
        self.classifier = NSFWClassifier()
        self.detector = ViolationDetector()
        
        self.segmenter = ViolationSegmenter()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, image: Image.Image) -> Dict[str, Any]:
        cls_result = self.classifier.predict(image)
        possible_violation = cls_result.get("possible_violation", False)

        if not possible_violation:
            filename = self._save_image(image)
            return {"status": "SAFE", "filename": filename, "violations": []}

        detections = self.detector.predict(image)
        violating_boxes = self._filter_violations(detections)

        if not violating_boxes:
            filename = self._save_image(image)
            return {"status": "SAFE_AFTER_DETECTION", "filename": filename, "violations": []}

        height, width = image.size[1], image.size[0]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        labels: List[str] = []

        for det in violating_boxes:
            labels.append(det["label"])
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            mask = self.segmenter.segment(image, bbox)
            full_mask = np.maximum(full_mask, mask)

        if not np.any(full_mask):
            filename = self._save_image(image)
            return {"status": "SAFE_NO_MASK", "filename": filename, "violations": []}

        blurred_image = blur_with_mask(image, full_mask)
        filename = self._save_image(blurred_image)

        return {
            "status": "VIOLATION_BLURRED",
            "filename": filename,
            "violations": sorted(set(labels)),
        }

    @staticmethod
    def _filter_violations(detections):
        violating_labels = {"nudity_explicit", "cigarette", "weapon", "smoking"}
        threshold = 0.5
        out = []
        for d in detections:
            if d.get("label") in violating_labels and d.get("score", 0) >= threshold:
                out.append(d)
        return out

    def _save_image(self, image: Image.Image) -> str:
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(self.output_dir, filename)
        image.save(path)
        return filename
