"""
Classical thermal-image candidate detection.

No AI, machine-learning model, pretrained weights, or network download is used.
The output is a list of regions that may contain a person; it is NOT a
semantic person classifier. Thresholds and geometry are heuristic and must
be validated against labelled real thermal-camera data.
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


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
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = to_gray(image)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def detect_people(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.35,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Return geometric/thermal candidates, not confirmed people."""
        gray = self.preprocess(image)
        height, width = gray.shape
        total = height * width
        if total == 0:
            return [], "classical-threshold-contours"

        # Hot foreground. A high percentile prevents a global fixed threshold
        # from assuming one camera palette or ambient temperature.
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        percentile = float(np.percentile(gray, 88))
        threshold = int(np.clip(max(otsu, percentile), 1, 254))
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Close small gaps, then remove isolated noise.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            ratio = area / total
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 3 or box_h < 5:
                continue
            aspect = box_h / float(box_w)
            if not 0.45 <= aspect <= 6.0:
                continue
            extent = area / float(box_w * box_h)
            if extent < 0.12:
                continue

            roi = gray[y:y + box_h, x:x + box_w]
            # A ranking heuristic only; never call this a probability.
            geometry_score = min(aspect / 2.0, 1.0)
            contrast_score = float(np.clip((np.mean(roi) - np.mean(gray)) / 80.0, 0, 1))
            score = 0.55 * geometry_score + 0.25 * extent + 0.20 * contrast_score
            candidates.append({
                "x": int(x), "y": int(y), "w": int(box_w), "h": int(box_h),
                "score": round(float(score), 3),
                "area_px": int(area),
                "aspect_ratio": round(float(aspect), 3),
                "mean_intensity": round(float(np.mean(roi)), 2),
                "candidate_type": "thermal_geometric_candidate",
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates, "classical-threshold-contours"

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
