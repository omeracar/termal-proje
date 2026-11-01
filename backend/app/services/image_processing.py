"""
Hassas termal insan tespit servisi
Küçük / uzaktaki insanlar için daha toleranslı
"""

import cv2
import numpy as np
from typing import List, Dict, Any


class ImageProcessingService:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2].astype(np.float32)
            red = image[:, :, 2].astype(np.float32)
            # Sıcak bölgeleri daha fazla vurgula
            combined = 0.4 * value + 0.6 * red
            gray = np.clip(combined, 0, 255).astype(np.uint8)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced

    def detect_human_candidates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        processed = self.preprocess(image)
        h, w = processed.shape[:2]
        total = h * w

        # Daha düşük eşik (küçük insanları kaçırmamak için)
        otsu_val, _ = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(int(otsu_val * 0.70), 35)   # daha düşük
        _, binary = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)

        # Morfoloji (daha az agresif)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Daha toleranslı alan aralığı
            if area < total * 0.0015 or area > total * 0.40:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / float(bw) if bw > 0 else 0

            # İnsan şekli
            if aspect < 1.1 or aspect > 5.5:
                continue

            # Minimum yükseklik (daha düşük)
            if bh < h * 0.06:
                continue

            # Doluluk
            extent = area / (bw * bh) if (bw * bh) > 0 else 0
            if extent < 0.18:
                continue

            # Skor
            score = (area / total) * 0.45 + min(aspect / 2.8, 1.0) * 0.35 + extent * 0.20

            candidates.append({
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": float(area),
                "aspect_ratio": round(aspect, 2),
                "extent": round(extent, 2),
                "score": round(score, 3)
            })

        # Skora göre sırala
        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return candidates[:6]

    def draw_detections(self, image: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        for i, cand in enumerate(candidates):
            x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"Aday {i+1}", (x, max(y-8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return result
