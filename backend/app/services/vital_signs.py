"""Keşif amaçlı periyodik termal sinyal analizi; tıbbi bir test ya da yaşam durumu testi değildir."""
from typing import Any, Dict, List
import numpy as np
from scipy.signal import detrend, find_peaks, welch


FINDING_COMPATIBLE = "Yaşam belirtisiyle uyumlu bulgu tespit edildi."
FINDING_UNDETERMINED = "Yaşam belirtisi belirlenemedi."
FINDING_INSUFFICIENT = "Analiz için yeterli veri bulunmuyor."


def with_finding(result: Dict[str, Any]) -> Dict[str, Any]:
    """Durum koduna şartnamedeki üç sonuç ifadesinden birini ekler."""
    status = result.get("status")
    if status == "periodic_signal_detected":
        result["finding"] = FINDING_COMPATIBLE
    elif status == "not_detected":
        result["finding"] = FINDING_UNDETERMINED
    else:
        result["finding"] = FINDING_INSUFFICIENT
    return result


class VitalSignsService:
    def analyze_temperature_series(
        self, temperature_series: List[float], fps: float = 30.0
    ) -> Dict[str, Any]:
        return with_finding(self._analyze(temperature_series, fps))

    def _analyze(self, temperature_series: List[float], fps: float) -> Dict[str, Any]:
        values = np.asarray(temperature_series, dtype=np.float64)
        if fps <= 0 or values.ndim != 1 or not np.all(np.isfinite(values)):
            return {"status": "insufficient_data", "message": "Örnekleme hızı veya sinyal geçersiz"}
        duration = len(values) / fps
        if len(values) < max(60, int(fps * 10)) or duration < 10:
            return {
                "status": "insufficient_data",
                "message": "En az 10 saniyelik, düzenli örneklenmiş geçerli video verisi gerekir.",
            }
        # İkinci dereceden polinom çıkarılır: kameranın yavaş, doğrusal olmayan parlaklık kaymaları
        # (otomatik kazanç vb.) solunum bandı (>= 0,1 Hz) dışında kalır ve sinyali bozmasın.
        time_axis = np.linspace(-1.0, 1.0, len(values))
        signal = values - np.polyval(np.polyfit(time_axis, values, 2), time_axis)
        if float(np.std(signal)) < 1e-6:
            return {
                "status": "not_detected",
                "message": "Ölçülebilir periyodik termal değişim saptanmadı.",
                "details": {"duration_s": duration, "sampling_rate_hz": fps},
            }

        frequencies, power = welch(signal, fs=fps, nperseg=min(len(signal), max(64, int(fps * 20))))
        # Solunum bandı; fiziksel olarak en az 3 tam döngü gözlenmiş olmalı.
        band = (frequencies >= 0.10) & (frequencies <= 0.70) & (frequencies * duration >= 3.0)
        if np.count_nonzero(band) < 2:
            return {"status": "insufficient_data", "message": "Frekans çözünürlüğü yetersiz."}
        band_indices = np.where(band)[0]
        peak_index = int(band_indices[np.argmax(power[band_indices])])
        peak_power = float(power[peak_index])
        peak_frequency = float(frequencies[peak_index])
        # Welch bin genişliği (~0,05 Hz = 3 nefes/dk) hız tahmini için kaba kalır: log-güç üzerinde
        # parabolik ara değerleme ile tepe konumu bin altı çözünürlüğe inceltilir (karar eşiğini etkilemez).
        if 0 < peak_index < len(frequencies) - 1 and min(power[peak_index - 1:peak_index + 2]) > 0:
            lo, mid, hi = np.log(power[peak_index - 1:peak_index + 2])
            denom = lo - 2 * mid + hi
            if denom < 0:
                shift = float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5))
                peak_frequency += shift * float(frequencies[1] - frequencies[0])

        # Tepe, sinyalin kendi (1/f biçimli) arka plan eğrisiyle kıyaslanır. Eğri, tepe çevresi
        # hariç log-log doğrusal uyumla bulunur; böylece kamera kayması ve rastgele sürüklenme
        # gibi düşük frekanslı gürültü yanlışlıkla periyodik bileşen sayılmaz.
        analysis_range = (frequencies >= 0.05) & (frequencies <= min(2.0, 0.95 * fps / 2.0))
        near_peak = np.abs(np.arange(len(frequencies)) - peak_index) <= 2
        use = analysis_range & ~near_peak & (power > 0)
        if np.count_nonzero(use) < 4 or peak_power <= 0:
            return {"status": "insufficient_data", "message": "Arka plan eğrisi için yeterli frekans noktası yok."}
        design = np.vstack([np.log(frequencies[use]), np.ones(np.count_nonzero(use))]).T
        slope, intercept = np.linalg.lstsq(design, np.log(power[use]), rcond=None)[0]
        trend_power = float(np.exp(slope * np.log(peak_frequency) + intercept))
        peak_to_trend_db = float(10 * np.log10(peak_power / max(trend_power, 1e-30)))
        # Eşikler, beyaz/pembe gürültü ve rastgele sürüklenme üzerinde yapılan simülasyonla
        # yanlış olumlu oranı yaklaşık %1-2 olacak biçimde seçildi (bkz. tests/test_vital_signs_false_positive.py).
        threshold_db = 14.0 if duration < 40.0 else 10.0
        rate_per_min = peak_frequency * 60.0
        detected = peak_to_trend_db >= threshold_db
        return {
            "status": "periodic_signal_detected" if detected else "not_detected",
            "message": (
                "Solunum hızı bandıyla uyumlu periyodik bir termal bileşen saptandı; bu, nefes alındığının veya yaşamın kanıtı değildir."
                if detected else
                "Solunum bandında yeterince belirgin periyodik termal bileşen saptanmadı; bu, yaşamın olmadığını göstermez."
            ),
            "details": {
                "duration_s": round(duration, 2),
                "sampling_rate_hz": fps,
                "dominant_frequency_hz": round(peak_frequency, 4),
                "respiratory_band_rate_per_min": round(rate_per_min, 1),
                "peak_to_trend_db": round(peak_to_trend_db, 2),
                "algorithm": "detrend+Welch PSD+1/f trend",
                "threshold_db": threshold_db,
            },
            "limitations": [
                "Yalnızca keşif amaçlı sinyal analizidir; doğrulanmış bir yaşam belirtisi algılayıcısı değildir.",
                "Sabit kamera geometrisi, yeterli kare hızı, uygun bir ilgi bölgesi ve gerçek veriyle doğrulama gerektirir.",
                "Hareket, hava akımı, örtülme ve kameranın otomatik aralık ayarı periyodik sinyal oluşturabilir ya da bastırabilir.",
                "Olumlu ya da olumsuz sonuç hiçbir zaman yaşamın veya ölümün doğrulaması olarak yorumlanmamalıdır.",
            ],
        }
