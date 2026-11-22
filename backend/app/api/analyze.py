import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.image_processing import ImageProcessingService
from app.services.temperature_analysis import TemperatureAnalysisService
from app.services.video_analysis import ThermalVideoAnalysisService

router = APIRouter(prefix="/api/v1", tags=["Analyze"])
processor = ImageProcessingService()
temperature_service = TemperatureAnalysisService()
video_processor = ThermalVideoAnalysisService()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024


def _run_analysis(image: np.ndarray):
    candidates, method = processor.detect_people(image)
    annotated = processor.draw_detections(image, candidates)
    measurements = [temperature_service.analyze_region(image, item) for item in candidates]
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode result image")
    return candidates, method, measurements, encoded


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Use JPG, PNG, TIFF or BMP for image analysis")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image upload size is 25 MB")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded")
    try:
        candidates, method, measurements, encoded = await run_in_threadpool(_run_analysis, image)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    annotated_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "message": "Analysis completed", "original_filename": file.filename,
        "image_shape": list(image.shape), "method": method,
        "detection_semantics": "thermal_candidate_regions_not_confirmed_people",
        "person_count": None, "candidate_count": len(candidates),
        "candidates": candidates, "temperature_analysis": measurements,
        "temperature_warning": "Pixel levels are DN, not °C. A calibrated radiometric temperature matrix is required for physical temperature values.",
        "vital_signs": {"status": "insufficient_data",
                        "message": "A single image cannot support temporal respiratory-signal analysis."},
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "num_candidates": len(candidates),
    }


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Use MP4, AVI, MOV or MKV for video analysis")
    contents = await file.read(MAX_VIDEO_BYTES + 1)
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Maximum video upload size is 200 MB")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        result = await run_in_threadpool(video_processor.analyze, temp_path)
        return {"message": "Video analysis completed", "original_filename": file.filename, **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
