"""Temperature statistics with strict separation of radiometric values and display pixels."""
from typing import Any, Dict, Optional
import numpy as np


class TemperatureAnalysisService:
    @staticmethod
    def _roi(array: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        x, y, w, h = (int(region[k]) for k in ("x", "y", "w", "h"))
        roi = array[y:y + h, x:x + w]
        if roi.size == 0:
            raise ValueError("The selected region is empty")
        return roi.astype(np.float64)

    def analyze_region(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        temperature_map_c: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Only temperature_map_c may be reported in °C; raw image levels are not temperatures."""
        if temperature_map_c is not None:
            if temperature_map_c.shape[:2] != image.shape[:2]:
                raise ValueError("Temperature map dimensions must match the image")
            values = self._roi(temperature_map_c, region)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError("No valid radiometric samples in region")
            return {
                "measurement_type": "radiometric_temperature",
                "unit": "degC",
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "sample_count": int(values.size),
                "calibration_provided": True,
            }

        values = self._roi(image, region)
        return {
            "measurement_type": "raw_pixel_intensity",
            "unit": "DN",
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "sample_count": int(values.size),
            "calibration_provided": False,
            "warning": "Pixel intensity is not a temperature. A calibrated radiometric temperature matrix is required for °C.",
        }

    def compare_with_background(
        self,
        temperature_map_c: np.ndarray,
        region: Dict[str, Any],
        margin: int = 8,
    ) -> Dict[str, Any]:
        """Compare a candidate ROI with a surrounding ring in a calibrated °C map."""
        x, y, w, h = (int(region[k]) for k in ("x", "y", "w", "h"))
        height, width = temperature_map_c.shape[:2]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
        outer = temperature_map_c[y0:y1, x0:x1].astype(np.float64).copy()
        inner = temperature_map_c[y:y + h, x:x + w]
        outer[y - y0:y - y0 + h, x - x0:x - x0 + w] = np.nan
        background = outer[np.isfinite(outer)]
        foreground = inner[np.isfinite(inner)]
        if not background.size or not foreground.size:
            return {"available": False, "reason": "Insufficient valid pixels"}
        fg, bg = float(np.mean(foreground)), float(np.mean(background))
        return {
            "available": True, "unit": "degC",
            "candidate_mean_c": fg, "background_mean_c": bg,
            "difference_c": fg - bg,
        }
