import os
import io
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from app.core import ModerationPipeline

try:
    from app.models.detector import get_detector
    _HAS_GET_DETECTOR = True
except Exception:
    get_detector = None
    _HAS_GET_DETECTOR = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-moderation")

app = FastAPI(
    title="Image Moderation + Segmentation API",
    version="0.1.0",
)

pipeline = ModerationPipeline(output_dir="safe_images")


def _try_load_detector_from_env():
    if not _HAS_GET_DETECTOR:
        logger.info("get_detector() not found in app.models.detector — skipping detector load.")
        return
    
    candidates = []
    env_path = os.environ.get("DETECTOR_WEIGHTS")
    if env_path:
        candidates.append(env_path)

    candidates.append(os.path.join("app", "models", "weights", "detector_best.pt"))

    candidates.append(os.path.join("ml", "weights", "detector_best.pt"))
    candidates.append(os.path.join("ml", "weights", "my_exported_best.pt"))

    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.exists(p):
            logger.info(f"[Detector] Found weights at: {p} — attempting to load.")
            try:
                det = get_detector(weights_path=p, device=None)  
                if getattr(det, "ready", False):
                    pipeline.detector = det
                    logger.info(f"[Detector] Loaded detector from: {p}")
                else:
                    pipeline.detector = det
                    logger.warning(f"[Detector] Detector instance created but not ready (check ultralytics install).")
                return
            except Exception as e:
                logger.exception(f"[Detector] Failed to load detector weights from {p}: {e}")
        else:
            logger.debug(f"[Detector] Weights not found at {p}")
    logger.info("[Detector] No detector weights loaded — using default pipeline detector (may be dummy).")


_try_load_detector_from_env()


@app.post("/moderate-image")
async def moderate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = pipeline.run(image)
    return JSONResponse(result)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector_ready": getattr(pipeline.detector, "ready", False),
        "detector_weights": getattr(pipeline.detector, "weights_path", None),
    }