"""Radyometrik değerler ile görüntüleme piksellerini kesin biçimde ayıran sıcaklık istatistikleri."""
from typing import Any, Dict, Optional
import numpy as np


class TemperatureAnalysisService:
    @staticmethod
    def _roi(array: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        x, y, w, h = (int(region[k]) for k in ("x", "y", "w", "h"))
        roi = array[y:y + h, x:x + w]
        if roi.size == 0:
            raise ValueError("Seçilen bölge boş")
        return roi.astype(np.float64)

    def analyze_region(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        temperature_map_c: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Yalnızca temperature_map_c °C olarak raporlanabilir; ham görüntü seviyeleri sıcaklık değildir."""
        if temperature_map_c is not None:
            if temperature_map_c.shape[:2] != image.shape[:2]:
                raise ValueError("Sıcaklık haritasının boyutları görüntüyle aynı olmalıdır")
            values = self._roi(temperature_map_c, region)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError("Bölgede geçerli radyometrik örnek yok")
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
            "warning": "Piksel yoğunluğu sıcaklık değildir. °C değeri için kalibre edilmiş radyometrik sıcaklık matrisi gerekir.",
        }

    def compare_with_background(
        self,
        temperature_map_c: np.ndarray,
        region: Dict[str, Any],
        margin: int = 8,
    ) -> Dict[str, Any]:
        """Kalibre edilmiş °C haritasında aday bölgeyi çevresindeki halkayla karşılaştırır."""
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
            return {"available": False, "reason": "Yeterli geçerli piksel yok"}
        fg, bg = float(np.mean(foreground)), float(np.mean(background))
        return {
            "available": True, "unit": "degC",
            "candidate_mean_c": fg, "background_mean_c": bg,
            "difference_c": fg - bg,
        }

    def compare_with_surroundings(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        temperature_map_c: Optional[np.ndarray] = None,
        margin: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aday bölgenin ortalamasını çevresindeki halkayla karşılaştırır.

        Kalibre edilmiş °C haritası verilirse sonuç °C cinsindendir; verilmezse ham gri seviye (DN)
        ile çalışılır ve fark yalnızca GÖRECE bir değerdir (fiziksel sıcaklık farkı değildir).
        """
        x, y, w, h = (int(region[k]) for k in ("x", "y", "w", "h"))
        if temperature_map_c is not None:
            array, unit, measurement = temperature_map_c, "degC", "radiometric_temperature"
        else:
            array, unit, measurement = image, "DN", "raw_pixel_intensity"
        if array.ndim == 3:
            array = array.mean(axis=2)
        if margin is None:
            margin = max(8, int(0.3 * max(w, h)))
        height, width = array.shape[:2]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
        outer = array[y0:y1, x0:x1].astype(np.float64).copy()
        inner = array[y:y + h, x:x + w].astype(np.float64)
        outer[y - y0:y - y0 + h, x - x0:x - x0 + w] = np.nan
        background = outer[np.isfinite(outer)]
        foreground = inner[np.isfinite(inner)]
        if not background.size or not foreground.size:
            return {"available": False, "reason": "Yeterli geçerli piksel yok"}
        fg, bg = float(np.mean(foreground)), float(np.mean(background))
        result = {
            "available": True, "measurement_type": measurement, "unit": unit,
            "candidate_mean": round(fg, 2), "background_mean": round(bg, 2),
            "difference": round(fg - bg, 2),
            "background_pixel_count": int(background.size),
        }
        if unit == "DN":
            result["note"] = "Fark görecedir (DN); fiziksel sıcaklık farkı değildir."
        return result

    def region_distribution(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        temperature_map_c: Optional[np.ndarray] = None,
        bins: int = 16,
    ) -> Dict[str, Any]:
        """Bölge içi değer dağılımı: histogram ve yüzdelikler (°C ya da görece DN)."""
        array = temperature_map_c if temperature_map_c is not None else image
        values = self._roi(array, region)
        values = values[np.isfinite(values)].astype(np.float64)
        if values.size == 0:
            return {"available": False, "reason": "Yeterli geçerli piksel yok"}
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
        p10, p50, p90 = (float(v) for v in np.percentile(values, [10, 50, 90]))
        return {
            "available": True,
            "unit": "degC" if temperature_map_c is not None else "DN",
            "bin_edges": [round(float(e), 2) for e in edges],
            "counts": [int(c) for c in counts],
            "p10": round(p10, 2), "p50": round(p50, 2), "p90": round(p90, 2),
        }
