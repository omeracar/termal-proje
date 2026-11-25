"""Keşif amaçlı periyodik termal sinyal analizi; tıbbi bir test ya da yaşam durumu testi değildir."""
from typing import Any, Dict, List
import numpy as np
from scipy.signal import detrend, find_peaks, welch


class VitalSignsService:
    def analyze_temperature_series(
        self, temperature_series: List[float], fps: float = 30.0
    ) -> Dict[str, Any]:
        values = np.asarray(temperature_series, dtype=np.float64)
        if fps <= 0 or values.ndim != 1 or not np.all(np.isfinite(values)):
            return {"status": "insufficient_data", "message": "Örnekleme hızı veya sinyal geçersiz"}
        duration = len(values) / fps
        if len(values) < max(60, int(fps * 10)) or duration < 10:
            return {
                "status": "insufficient_data",
                "message": "En az 10 saniyelik, düzenli örneklenmiş geçerli video verisi gerekir.",
            }
        signal = detrend(values, type="linear")
        if float(np.std(signal)) < 1e-6:
            return {
                "status": "not_detected",
                "message": "Ölçülebilir periyodik termal değişim saptanmadı.",
                "details": {"duration_s": duration, "sampling_rate_hz": fps},
            }

        frequencies, power = welch(signal, fs=fps, nperseg=min(len(signal), max(64, int(fps * 20))))
        band = (frequencies >= 0.10) & (frequencies <= 0.70)
        if np.count_nonzero(band) < 3:
            return {"status": "insufficient_data", "message": "Frekans çözünürlüğü yetersiz."}
        band_power = power[band]
        band_freq = frequencies[band]
        peak_index = int(np.argmax(band_power))
        peak_power = float(band_power[peak_index])
        noise_floor = float(np.median(power[~band])) if np.any(~band) else 0.0
        snr_db = float(10 * np.log10((peak_power + 1e-15) / (noise_floor + 1e-15)))
        peak_frequency = float(band_freq[peak_index])
        rate_per_min = peak_frequency * 60.0
        detected = snr_db >= 6.0 and peak_power > 0
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
                "peak_to_median_snr_db": round(snr_db, 2),
                "algorithm": "detrend+Welch PSD",
                "snr_threshold_db": 6.0,
            },
            "limitations": [
                "Yalnızca keşif amaçlı sinyal analizidir; doğrulanmış bir yaşam belirtisi algılayıcısı değildir.",
                "Sabit kamera geometrisi, yeterli kare hızı, uygun bir ilgi bölgesi ve gerçek veriyle doğrulama gerektirir.",
                "Hareket, hava akımı, örtülme ve kameranın otomatik aralık ayarı periyodik sinyal oluşturabilir ya da bastırabilir.",
                "Olumlu ya da olumsuz sonuç hiçbir zaman yaşamın veya ölümün doğrulaması olarak yorumlanmamalıdır.",
            ],
        }
