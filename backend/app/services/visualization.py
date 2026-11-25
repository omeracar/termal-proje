"""Sıcaklık/yoğunluk dağılımının görselleştirilmesi (yalnızca OpenCV ve NumPy)."""
from typing import Any, Dict, List

import cv2
import numpy as np

from app.services.image_processing import to_gray


def build_heatmap(image: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    """Gri seviyeyi renk haritasına çevirir, aday kutularını çizer, sağa renk skalası ekler.

    Renkler GÖRECE yoğunluğu gösterir (koyu mor = düşük, sarı/beyaz = yüksek). Kalibre edilmiş
    radyometrik veri olmadan bu renkler °C anlamına gelmez; skalanın üst/alt etiketlerini
    arayüz yazar.
    """
    gray = to_gray(image)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    for item in candidates:
        x, y, w, h = (int(item[k]) for k in ("x", "y", "w", "h"))
        cv2.rectangle(colored, (x, y), (x + w, y + h), (255, 255, 255), 2)
    height = colored.shape[0]
    bar_w = max(14, colored.shape[1] // 25)
    ramp = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(ramp, bar_w, axis=1), cv2.COLORMAP_INFERNO)
    spacer = np.zeros((height, 6, 3), dtype=np.uint8)
    return np.hstack([colored, spacer, bar])
