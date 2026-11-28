"""Solunum doğrulama hattı testleri.

NOT: Sentetik veriyle yapılan testler yalnızca KOD DOĞRULUĞUNU sınar (hat kendi içinde tutarlı mı);
algılayıcının gerçek kamera verisindeki başarımının kanıtı değildir.
"""
import datetime as dt

import numpy as np

from app.services.breathing_validation import (
    auto_roi, celsius_from_raw, parse_event_times, reference_rate_per_min,
    resample_uniform, roi_series, validate_breathing,
)

T0 = dt.datetime(2016, 11, 22, 14, 23, 9, tzinfo=dt.timezone.utc).timestamp()


def _csv(event_offsets, duration=120.0, step=0.04):
    lines = []
    t = 0.0
    while t < duration:
        on = any(0 <= t - e < 0.045 for e in event_offsets)
        stamp = dt.datetime.fromtimestamp(T0 + t, dt.timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        lines.append(f"{stamp},{1 if on else 0}")
        t += step
    return "\n".join(lines) + "\n"


def test_parse_event_times_counts_rising_edges_only():
    events = parse_event_times(_csv([2.0, 5.0, 8.1]))
    assert len(events) == 3
    assert np.allclose(events - T0, [2.0, 5.0, 8.1], atol=0.06)


def test_parse_skips_blank_and_bad_rows():
    text = "20161122-14:23:09.000,0\n\nbozuk,satir\n20161122-14:23:09.100,1\n20161122-14:23:09.200,\n20161122-14:23:09.300,0\n"
    assert len(parse_event_times(text)) == 1


def test_reference_rate_from_median_interval():
    events = np.arange(0, 60, 3.0) + T0  # 20 nefes/dk
    assert abs(reference_rate_per_min(events, T0, T0 + 40) - 20.0) < 1e-6
    assert reference_rate_per_min(events, T0, T0 + 8) is None  # yetersiz olay


def test_celsius_conversion_matches_dataset_formula():
    assert celsius_from_raw(np.array([1340]))[0] == 34.0


def test_resample_uniform_recovers_rate_from_jittered_timestamps():
    rng = np.random.default_rng(0)
    t = np.cumsum(np.full(400, 0.1) + rng.normal(0, 0.003, 400))
    grid, values, fps = resample_uniform(t, np.sin(t))
    assert abs(fps - 10.0) < 0.2 and np.all(np.diff(grid) > 0)


def test_auto_roi_finds_warm_blob():
    frames = np.full((20, 40, 60), 25.0)
    frames[:, 10:22, 30:42] = 35.0
    x, y, w, h = auto_roi(frames)
    assert 28 <= x <= 32 and 8 <= y <= 12 and 10 <= w <= 14 and 10 <= h <= 14
    assert roi_series(frames, (x, y, w, h)).mean() > 34


def test_pipeline_recovers_known_breathing_rate_on_synthetic_data():
    period = 3.2  # 18,75 nefes/dk
    times = T0 + np.arange(0, 120, 0.1)
    rng = np.random.default_rng(5)
    series = 34 + 0.15 * np.sin(2 * np.pi * (times - T0) / period) + rng.normal(0, 0.05, times.size)
    events = parse_event_times(_csv(list(np.arange(0.5, 120, period))))
    result = validate_breathing(times, series, events)
    s = result["summary"]
    assert s["windows_with_reference"] >= 4
    assert s["detection_rate"] >= 0.75
    assert s["mean_abs_error_per_min"] <= 1.5
    assert s["within_tolerance_fraction"] >= 0.75


def test_pipeline_does_not_claim_breathing_in_noise():
    rng = np.random.default_rng(9)
    times = T0 + np.arange(0, 120, 0.1)
    series = 34 + np.cumsum(rng.normal(0, 0.01, times.size))
    events = parse_event_times(_csv(list(np.arange(0.5, 120, 3.2))))
    result = validate_breathing(times, series, events)
    assert result["summary"]["detection_rate"] <= 0.25


def test_offset_shifts_reference_windows():
    times = T0 + np.arange(0, 120, 0.1)
    series = 34 + 0.1 * np.sin(2 * np.pi * (times - T0) / 3.0)
    events = np.arange(0, 120, 3.0) + T0 - 500  # saat farkı: 500 sn geride
    none = validate_breathing(times, series, events)
    fixed = validate_breathing(times, series, events, offset_s=500.0)
    assert none["summary"]["windows_with_reference"] == 0
    assert fixed["summary"]["windows_with_reference"] >= 4
