from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

from app.services.image_processing import ImageProcessingService

router = APIRouter(prefix="/api/v1", tags=["Analyze"])

processor = ImageProcessingService()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Yüklenen termal görüntüyü analiz eder.
    İnsan olabilecek bölgeleri tespit eder ve sonuçları döner.
    """
    # Dosya uzantısı kontrolü
    allowed = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Sadece görüntü dosyaları destekleniyor")

    # Dosyayı oku
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")

    # Tespit
    candidates = processor.detect_human_candidates(image)

    # Sonuç görüntüsü oluştur
    result_image = processor.draw_detections(image, candidates)

    # JPEG olarak encode et
    success, encoded = cv2.imencode(".jpg", result_image)
    if not success:
        raise HTTPException(status_code=500, detail="Sonuç görüntüsü oluşturulamadı")

    return {
        "message": "Analiz tamamlandı",
        "original_filename": file.filename,
        "image_shape": list(image.shape),
        "num_candidates": len(candidates),
        "candidates": candidates,
        # Not: Görüntüyü base64 olarak da dönebiliriz, şimdilik sadece metadata
    }
