"""
Tespit edilen bölgeler üzerinde sıcaklık analizi yapar.
"""

import numpy as np
from typing import Dict, Any, Optional


class TemperatureAnalysisService:
    def __init__(self):
        pass

    def analyze_region(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bir bölgenin sıcaklık istatistiklerini hesaplar.
        image: termal görüntü (radyometrik veya normalize edilmiş)
        region: {"x":, "y":, "w":, "h":}
        """
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return {}

        stats = {
            "mean": float(np.mean(roi)),
            "min": float(np.min(roi)),
            "max": float(np.max(roi)),
            "std": float(np.std(roi)),
        }
        return stats

    def compare_with_background(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """Bölge sıcaklığını çevre ile karşılaştırır"""
        # TODO
        return {}
