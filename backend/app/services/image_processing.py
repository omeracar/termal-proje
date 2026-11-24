"""
Classical thermal-image candidate detection.

No AI, machine-learning model, pretrained weights, or network download is used.
The output is a list of regions that may contain a person; it is NOT a
semantic person classifier. Thresholds and geometry are heuristic and must
be validated against labelled real thermal-camera data (see scripts/evaluate_detector.py).
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.services.classical_detector import ClassicalPersonDetector

METHOD_NAME = "classical-multilevel-contours"


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel 8-bit analysis image."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if gray.dtype == np.uint8:
        return gray
    values = gray.astype(np.float32)
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def to_bgr8(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class ImageProcessingService:
    def __init__(self):
        self.detector = ClassicalPersonDetector()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Contrast-enhanced 8-bit image (kept for visual inspection / debugging)."""
        gray = to_gray(image)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def detect_people(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
        """Return geometric/thermal candidates, not confirmed people."""
        gray = to_gray(image)
        boxes = self.detector.detect(gray)

        candidates: List[Dict[str, Any]] = []
        for box in boxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            roi = gray[y:y + h, x:x + w]
            candidates.append({
                "x": x, "y": y, "w": w, "h": h,
                # A ranking heuristic only; never call this a probability.
                "score": round(float(box["score"]), 3),
                "area_px": int(w * h),
                "aspect_ratio": round(h / float(w), 3),
                "mean_intensity": round(float(np.mean(roi)), 2),
                "candidate_type": "thermal_geometric_candidate",
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates, METHOD_NAME

    def draw_detections(self, image: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
        result = to_bgr8(image).copy()
        for index, item in enumerate(candidates, start=1):
            x, y, w, h = (item[k] for k in ("x", "y", "w", "h"))
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(result, f"Aday {index}", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(result, f"Aday bolge: {len(candidates)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
        return result
