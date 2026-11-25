import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRecord
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
        raise RuntimeError("Sonuç görüntüsü kodlanamadı")
    return candidates, method, measurements, encoded


def _save_record(db: Session, **values) -> AnalysisRecord:
    record = AnalysisRecord(**values)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Görüntü analizi için JPG, PNG, TIFF veya BMP kullanın")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="En büyük görüntü yükleme boyutu 25 MB")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Görüntü çözülemedi")
    try:
        candidates, method, measurements, encoded = await run_in_threadpool(_run_analysis, image)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    vital = {"status": "insufficient_data",
             "message": "Tek bir görüntüyle zamansal solunum sinyali analizi yapılamaz."}
    result_data = {
        "image_shape": list(image.shape), "method": method,
        "detection_semantics": "thermal_candidate_regions_not_confirmed_people",
        "candidates": candidates, "temperature_analysis": measurements,
        "temperature_warning": "Piksel seviyeleri DN cinsindendir, °C değildir. Fiziksel sıcaklık için kalibre edilmiş radyometrik sıcaklık matrisi gerekir.",
        "vital_signs": vital,
    }
    try:
        record = _save_record(
            db, analysis_type="image", original_filename=(file.filename or "upload")[:255],
            candidate_count=len(candidates),
            summary={
                "candidate_region_count": len(candidates),
                "image_dimensions": " × ".join(str(v) for v in image.shape),
                "processing_method": method,
                "temperature_unit": "DN (raw pixel intensity)",
                "vital_signal_status": vital["status"],
            },
            result_data=result_data,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Analiz tamamlandı ancak veritabanına kaydedilemedi. PostgreSQL ayarlarını kontrol edin.") from exc

    annotated_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "analysis_id": record.id, "message": "Analiz tamamlandı ve kaydedildi",
        "original_filename": file.filename, **result_data,
        "person_count": None, "candidate_count": len(candidates),
        "temperature_analysis": measurements, "vital_signs": vital,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "num_candidates": len(candidates),
    }


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Video analizi için MP4, AVI, MOV veya MKV kullanın")
    contents = await file.read(MAX_VIDEO_BYTES + 1)
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="En büyük video yükleme boyutu 200 MB")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        result = await run_in_threadpool(video_processor.analyze, temp_path)
        duration = float(result["video"]["duration_s"])
        record = _save_record(
            db, analysis_type="video", original_filename=(file.filename or "upload")[:255],
            candidate_count=int(result.get("track_count", 0)),
            duration_seconds=int(duration),
            summary={
                "duration_seconds": duration,
                "source_fps": result["video"]["source_fps"],
                "sampled_frames": result["video"]["sampled_frames"],
                "candidate_track_count": result["track_count"],
                "tracking_method": result["tracking_method"],
                "signal_unit": "DN (raw grayscale intensity)",
            },
            result_data=result,
        )
        return {"analysis_id": record.id, "message": "Video analizi tamamlandı ve kaydedildi",
                "original_filename": file.filename, **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Video analizi kaydedilemedi. PostgreSQL ayarlarını kontrol edin.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
