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


def reference_time_range(source: "str | Path") -> Tuple[float, float, int]:
    """Referans CSV'sinin ilk/son satır zamanı (epoch saniye, UTC kabul) ve satır sayısı."""
    stamps = []
    for row in csv.reader(io.StringIO(Path(source).read_text(encoding="utf-8"))):
        if len(row) == 2 and row[1].strip():
            try:
                stamps.append(dt.datetime.strptime(row[0].strip(), "%Y%m%d-%H:%M:%S.%f")
                              .replace(tzinfo=dt.timezone.utc).timestamp())
            except ValueError:
                continue
    return (stamps[0], stamps[-1], len(stamps)) if stamps else (float("nan"), float("nan"), 0)


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
    return auto_roi_from_mean(frames_celsius.mean(axis=0), percentile)


def auto_roi_from_mean(mean: np.ndarray, percentile: float = 95.0) -> Tuple[int, int, int, int]:
    """auto_roi ile aynı, ancak hazır bir ortalama kareden (büyük kayıtlarda bellek tasarrufu)."""
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


def hot_track(raw: np.ndarray, chunk: int = 200, hot_fraction: float = 0.6):
    """Her karede en büyük sıcak bileşeni (baş/gövde) bulup ağırlık merkezini izler.

    Eşik, karenin ortancası ile %99,5'lik değeri arasında `hot_fraction` noktasıdır (sezgisel).
    Döndürür: (merkezler (n,2) [x,y], hiç-sıcak-olmayan-piksel için 'hiç sıcak oldu' maskesi (h,w)).
    """
    n, height, width = raw.shape
    centers = np.full((n, 2), np.nan)
    ever = np.zeros((height, width), dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    for start in range(0, n, chunk):
        block = raw[start:start + chunk].astype(np.float32)
        for j, frame in enumerate(block):
            sub = frame[::2, ::2]
            lo, hi = float(np.median(sub)), float(np.percentile(sub, 99.5))
            if hi - lo < 1e-6:
                continue
            mask = (frame >= lo + hot_fraction * (hi - lo)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if count <= 1:
                continue
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            centers[start + j] = cents[best]
            ever |= labels == best
    return centers, ever


def head_shifts(centers: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Merkezlerin ortanca konuma göre kaymaları (n,2); eksikler ara değerlenir, ortanca süzgeçle yumuşatılır."""
    from scipy.ndimage import median_filter
    out = np.empty_like(centers)
    idx = np.arange(len(centers))
    for axis in range(2):
        col = centers[:, axis]
        ok = np.isfinite(col)
        col = np.interp(idx, idx[ok], col[ok]) if ok.any() else np.zeros(len(col))
        out[:, axis] = median_filter(col - np.median(col), size=smooth, mode="nearest")
    return out


def tracked_roi_series(raw: np.ndarray, roi: Tuple[int, int, int, int], shifts: np.ndarray) -> np.ndarray:
    """ROI kutusunu her karede baş kaymasıyla birlikte oynatıp ortalama sıcaklığı (°C) döndürür."""
    x, y, w, h = roi
    n, height, width = raw.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x0 = int(np.clip(x + int(round(shifts[i, 0])), 0, width - w))
        y0 = int(np.clip(y + int(round(shifts[i, 1])), 0, height - h))
        out[i] = raw[i, y0:y0 + h, x0:x0 + w].mean(dtype=np.float64)
    return celsius_from_raw(out)


def static_background_series(raw: np.ndarray, ever_hot: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Hiçbir karede sıcak bileşene ait olmayan (genişletilmiş) piksellerin kare başına ortalaması (°C).

    Kameranın tüm görüntüyü etkileyen ani kayması (otomatik kalibrasyon vb.) bu seride de görünür.
    """
    dilated = cv2.dilate(ever_hot.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    mask = ~dilated
    if mask.sum() < 50:
        mask = ~ever_hot
    out = np.empty(len(raw), dtype=np.float64)
    for start in range(0, len(raw), chunk):
        out[start:start + chunk] = raw[start:start + chunk][:, mask].mean(axis=1, dtype=np.float64)
    return celsius_from_raw(out)


def count_steps(series: np.ndarray, fps: float, threshold_c: float = 0.4, window_s: float = 0.5) -> int:
    """`window_s` içinde `threshold_c`'den büyük sıçrama sayısı (1 sn içindeki sıçramalar tek sayılır)."""
    k = max(1, int(round(window_s * fps)))
    jump = np.abs(series[k:] - series[:-k]) > threshold_c
    count, last = 0, -10 ** 9
    for i in np.where(jump)[0]:
        if i - last > fps:
            count += 1
        last = i
    return count
