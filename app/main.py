from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.core import ModerationPipeline

app = FastAPI(
    title="Image Moderation + Segmentation API",
    version="0.1.0",
)

# single pipeline instance (models loaded once)
pipeline = ModerationPipeline(output_dir="safe_images")


@app.post("/moderate-image")
async def moderate_image(file: UploadFile = File(...)):
    
    # receives an uploaded file,moderates it using pipeline and returns result in JSON
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = pipeline.run(image)
    return JSONResponse(result)
