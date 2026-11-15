import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.image_processing import ImageProcessingService

router = APIRouter(prefix="/api/v1", tags=["Analyze"])

processor = ImageProcessingService()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _run_analysis(image: np.ndarray):
    people, method = processor.detect_people(image)
    annotated = processor.draw_detections(image, people)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Sonuç görüntüsü oluşturulamadı")
    return people, method, encoded


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Yüklenen termal görüntüdeki kişileri tespit eder, sayar ve
    kutularla işaretlenmiş sonuç görüntüsünü (base64) döner.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Sadece görüntü dosyaları destekleniyor")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")

    try:
        people, method, encoded = await run_in_threadpool(_run_analysis, image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    annotated_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    return {
        "message": "Analiz tamamlandı",
        "original_filename": file.filename,
        "image_shape": list(image.shape),
        "method": method,
        "person_count": len(people),
        "people": people,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        # Eski alan adları (geriye dönük uyumluluk)
        "num_candidates": len(people),
        "candidates": people,
    }
