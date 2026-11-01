from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["Upload"])

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".mp4", ".avi", ".mov"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Termal fotoğraf veya video yükleme endpoint'i"""
    
    # Uzantı kontrolü
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı. İzin verilenler: {ALLOWED_EXTENSIONS}"
        )

    # Benzersiz dosya adı
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    # Kaydet
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya kaydedilemedi: {str(e)}")

    return {
        "message": "Dosya başarıyla yüklendi",
        "filename": safe_name,
        "original_name": file.filename,
        "path": str(save_path),
        "size_bytes": save_path.stat().st_size
    }
