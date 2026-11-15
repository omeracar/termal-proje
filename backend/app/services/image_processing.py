"""
Termal görüntü işleme servisi.

Ana yöntem: YOLOX tabanlı insan tespiti (person_detector.py).
Model indirilemezse (ör. internet yok) eski eşikleme + kontur yöntemi yedek olarak kullanılır.
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.services.person_detector import ModelUnavailableError, PersonDetector, to_bgr8

logger = logging.getLogger(__name__)


class ImageProcessingService:
    def __init__(self):
        self.detector = PersonDetector()

    # ------------------------------------------------------------------
    # Ana tespit
    # ------------------------------------------------------------------
    def detect_people(
        self, image: np.ndarray, conf_threshold: float = 0.30
    ) -> Tuple[List[Dict[str, Any]], str]:
        """(kişi listesi, kullanılan yöntem) döner."""
        try:
            return self.detector.detect(image, conf_threshold=conf_threshold), "yolox"
        except ModelUnavailableError as exc:
            logger.warning("YOLOX kullanılamadı, eski yönteme geçildi: %s", exc)
            return self.detect_human_candidates(image), "threshold-legacy"

    # ------------------------------------------------------------------
    # Çizim
    # ------------------------------------------------------------------
    def draw_detections(self, image: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        """Her kişiyi numaralı kutuyla işaretler ve sol üste toplam sayıyı yazar."""
        result = to_bgr8(image).copy()
        h, w = result.shape[:2]

        scale = max(0.5, min(h, w) / 800)
        thickness = max(2, int(round(scale * 2)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 220, 0)  # BGR yeşil

        for i, p in enumerate(people, start=1):
            x, y, bw, bh = p["x"], p["y"], p["w"], p["h"]
            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, thickness)

            label = f"Kisi {i}"
            if "score" in p:
                label += f" %{int(round(p['score'] * 100))}"
            (tw, th), base = cv2.getTextSize(label, font, 0.5 * scale, 1)
            ty = y - 4 if y - th - base - 4 >= 0 else y + bh + th + 4
            cv2.rectangle(result, (x, ty - th - base), (x + tw + 4, ty + base), color, -1)
            cv2.putText(result, label, (x + 2, ty), font, 0.5 * scale, (0, 0, 0), 1, cv2.LINE_AA)

        banner = f"Kisi sayisi: {len(people)}"
        (tw, th), base = cv2.getTextSize(banner, font, 0.8 * scale, 2)
        cv2.rectangle(result, (0, 0), (tw + 16, th + base + 14), (0, 0, 0), -1)
        cv2.putText(result, banner, (8, th + 7), font, 0.8 * scale, color, 2, cv2.LINE_AA)
        return result

    # ------------------------------------------------------------------
    # Yedek yöntem (eski): eşikleme + kontur. Sıcak/soğuk kontrastı yüksek,
    # arka planı koyu sahneler için; FLIR sokak sahnelerinde yetersizdir.
    # ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = to_bgr8(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)
        combined = 0.4 * value + 0.6 * red
        gray = np.clip(combined, 0, 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(blurred)

    def detect_human_candidates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        processed = self.preprocess(image)
        h, w = processed.shape[:2]
        total = h * w

        otsu_val, _ = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(int(otsu_val * 0.70), 35)
        _, binary = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total * 0.0015 or area > total * 0.40:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / float(bw) if bw > 0 else 0
            if aspect < 1.1 or aspect > 5.5:
                continue
            if bh < h * 0.06:
                continue

            extent = area / (bw * bh) if (bw * bh) > 0 else 0
            if extent < 0.18:
                continue

            score = (area / total) * 0.45 + min(aspect / 2.8, 1.0) * 0.35 + extent * 0.20
            candidates.append({
                "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                "score": round(score, 3),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:6]
