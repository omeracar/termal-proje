"""Solunum algılayıcısının referans solunum kaydıyla doğrulanması (LCAS veri seti biçimi).

Referans dosyası: her satır `YYYYMMDD-SS:DD:SS.mmm,değer`; değer 0/1 olay işaretidir ve her nefeste
kısa bir "1" darbesi bulunur. Karşılaştırma pencere pencere, nefes/dakika cinsinden yapılır.
Bu modül yalnızca sinyal işleme ve istatistik kullanır (yapay zekâ yoktur).
"""
import csv
import datetime as dt
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.services.vital_signs import VitalSignsService


def celsius_from_raw(raw: np.ndarray, offset: float = 1000.0, scale: float = 10.0) -> np.ndarray:
    """LCAS/Optris dönüşümü: derece_C = (ham - 1000) / 10 (veri seti sayfasındaki formül)."""
    return (raw.astype(np.float64) - offset) / scale


def parse_event_times(source: "str | Path") -> np.ndarray:
    """Referans CSV'sinden nefes olayı zamanlarını (epoch saniye, UTC kabul edilir) döndürür.

    Olay, değerin 0'dan 1'e geçtiği satırdır. Boş/bozuk satırlar atlanır.
    """
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()) else str(source)
    times: List[float] = []
    values: List[int] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 2 or not row[1].strip():
            continue
        try:
            stamp = dt.datetime.strptime(row[0].strip(), "%Y%m%d-%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)
            value = int(float(row[1]))
        except ValueError:
            continue
        times.append(stamp.timestamp())
        values.append(value)
    if not times:
        return np.array([], dtype=np.float64)
    v = np.asarray(values)
    rising = np.where((v[1:] == 1) & (v[:-1] == 0))[0] + 1
    if v[0] == 1:
        rising = np.r_[0, rising]
    return np.asarray(times, dtype=np.float64)[rising]


def reference_rate_per_min(event_times: Sequence[float], start: float, end: float,
                           min_events: int = 4) -> Optional[float]:
    """Pencere içindeki olayların ortanca aralığından nefes/dakika (yetersizse None)."""
    inside = np.asarray([t for t in event_times if start <= t < end])
    if inside.size < min_events:
        return None
    return float(60.0 / np.median(np.diff(inside)))


def auto_roi(frames_celsius: np.ndarray, percentile: float = 95.0) -> Tuple[int, int, int, int]:
    """Ortalama karedeki en sıcak, en büyük bağlı bileşenin sınırlayıcı kutusu (x, y, w, h).

    Sezgiseldir: kameraya yakın tek kişilik kayıtlarda yüz/baş bölgesini yakalamayı hedefler.
    Doğruluğu gerçek kayıtta gözle doğrulanmalıdır.
    """
    mean = frames_celsius.mean(axis=0)
    mask = (mean >= np.percentile(mean, percentile)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        h, w = mean.shape
        return 0, 0, w, h
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(stats[best, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return x, y, w, h


def roi_series(frames_celsius: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frames_celsius[:, y:y + h, x:x + w].reshape(len(frames_celsius), -1).mean(axis=1)


def resample_uniform(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Düzensiz zaman damgalı seriyi, ortanca kare hızında düzenli ızgaraya doğrusal ara değerlemeyle taşır."""
    order = np.argsort(times)
    t, v = times[order], values[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, v = t[keep], v[keep]
    fps = 1.0 / float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], 1.0 / fps)
    return grid, np.interp(grid, t, v), fps


def validate_breathing(
    times: np.ndarray,
    series: np.ndarray,
    event_times: Sequence[float],
    window_s: float = 40.0,
    step_s: float = 20.0,
    offset_s: float = 0.0,
    tolerance_per_min: float = 3.0,
) -> Dict[str, Any]:
    """Pencere pencere tahmin edilen nefes hızını referansla karşılaştırır.

    `offset_s`: referans saatine eklenecek kaydırma (iki bilgisayarın saat farkı için).
    Yalnızca referansı olan pencereler hata ölçütüne girer. Algılanmayan pencerelerde de
    baskın frekanstan bir hız tahmini vardır; ayrıca raporlanır.
    """
    grid, uniform, fps = resample_uniform(np.asarray(times, dtype=np.float64), np.asarray(series, dtype=np.float64))
    service = VitalSignsService()
    events = np.asarray(event_times, dtype=np.float64) + offset_s
    windows: List[Dict[str, Any]] = []
    start = grid[0]
    while start + window_s <= grid[-1] + 1e-9:
        end = start + window_s
        mask = (grid >= start) & (grid < end)
        result = service.analyze_temperature_series(uniform[mask].tolist(), fps=fps)
        details = result.get("details") or {}
        est = details.get("respiratory_band_rate_per_min")
        ref = reference_rate_per_min(events, start, end)
        windows.append({
            "start_s": round(float(start - grid[0]), 1), "end_s": round(float(end - grid[0]), 1),
            "status": result["status"], "detected": result["status"] == "periodic_signal_detected",
            "estimated_per_min": est, "reference_per_min": None if ref is None else round(ref, 2),
            "abs_error_per_min": None if (est is None or ref is None) else round(abs(est - ref), 2),
            "peak_to_trend_db": details.get("peak_to_trend_db"),
        })
        start += step_s
    scored = [w for w in windows if w["abs_error_per_min"] is not None]
    detected_scored = [w for w in scored if w["detected"]]
    errors = [w["abs_error_per_min"] for w in scored]
    return {
        "sampling_rate_hz": round(fps, 3),
        "duration_s": round(float(grid[-1] - grid[0]), 1),
        "window_s": window_s, "step_s": step_s, "offset_s": offset_s,
        "windows": windows,
        "summary": {
            "windows_total": len(windows),
            "windows_with_reference": len(scored),
            "windows_detected": sum(w["detected"] for w in windows),
            "detection_rate": round(len(detected_scored) / len(scored), 3) if scored else None,
            "mean_abs_error_per_min": round(float(np.mean(errors)), 2) if errors else None,
            "median_abs_error_per_min": round(float(np.median(errors)), 2) if errors else None,
            "within_tolerance_fraction": round(float(np.mean([e <= tolerance_per_min for e in errors])), 3) if errors else None,
            "tolerance_per_min": tolerance_per_min,
        },
    }
