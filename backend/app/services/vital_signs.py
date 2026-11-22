"""Exploratory periodic thermal-signal analysis; not a medical or life-status test."""
from typing import Any, Dict, List
import numpy as np
from scipy.signal import detrend, find_peaks, welch


class VitalSignsService:
    def analyze_temperature_series(
        self, temperature_series: List[float], fps: float = 30.0
    ) -> Dict[str, Any]:
        values = np.asarray(temperature_series, dtype=np.float64)
        if fps <= 0 or values.ndim != 1 or not np.all(np.isfinite(values)):
            return {"status": "insufficient_data", "message": "Invalid sampling rate or signal"}
        duration = len(values) / fps
        if len(values) < max(60, int(fps * 10)) or duration < 10:
            return {
                "status": "insufficient_data",
                "message": "At least 10 seconds of valid, consistently sampled video data is required.",
            }
        signal = detrend(values, type="linear")
        if float(np.std(signal)) < 1e-6:
            return {
                "status": "not_detected",
                "message": "No measurable periodic thermal variation was detected.",
                "details": {"duration_s": duration, "sampling_rate_hz": fps},
            }

        frequencies, power = welch(signal, fs=fps, nperseg=min(len(signal), max(64, int(fps * 20))))
        band = (frequencies >= 0.10) & (frequencies <= 0.70)
        if np.count_nonzero(band) < 3:
            return {"status": "insufficient_data", "message": "Frequency resolution is insufficient."}
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
                "A periodic thermal component compatible with a respiratory-rate band was detected; this is not proof of breathing or life."
                if detected else
                "No sufficiently prominent respiratory-band periodic thermal component was detected; this does not establish absence of life."
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
                "Exploratory signal analysis only; not a validated vital-sign detector.",
                "Requires stable camera geometry, sufficient frame rate, a suitable ROI and ground-truth validation.",
                "Motion, airflow, occlusion and camera auto-ranging can create or suppress periodic signals.",
                "A positive or negative result must never be interpreted as confirmation of life or death.",
            ],
        }
