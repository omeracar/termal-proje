import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.image_processing import ImageProcessingService
from app.services.temperature_analysis import TemperatureAnalysisService

router = APIRouter(prefix="/api/v1", tags=["Analyze"])
processor = ImageProcessingService()
temperature_service = TemperatureAnalysisService()
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _run_analysis(image: np.ndarray):
    candidates, method = processor.detect_people(image)
    annotated = processor.draw_detections(image, candidates)
    measurements = [
        temperature_service.analyze_region(image, candidate)
        for candidate in candidates
    ]
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode result image")
    return candidates, method, measurements, encoded


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Detect classical thermal candidates; never infer life status from a still image."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, TIFF and BMP images are supported")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum upload size is 25 MB")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded")
    try:
        candidates, method, measurements, encoded = await run_in_threadpool(_run_analysis, image)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    annotated_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "message": "Analysis completed",
        "original_filename": file.filename,
        "image_shape": list(image.shape),
        "method": method,
        "detection_semantics": "thermal_candidate_regions_not_confirmed_people",
        "person_count": None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "temperature_analysis": measurements,
        "temperature_warning": "Displayed pixel levels are DN, not °C. Provide a calibrated radiometric temperature matrix for physical temperature values.",
        "vital_signs": {
            "status": "insufficient_data",
            "message": "A single image cannot support temporal respiratory-signal analysis.",
        },
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        # Backward-compatible fields. The count refers to candidate regions.
        "num_candidates": len(candidates),
    }
