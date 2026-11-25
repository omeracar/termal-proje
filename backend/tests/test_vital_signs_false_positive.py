"""Solunum bandı algılayıcısının yanlış olumlu oranı ve duyarlılığı (sabit tohumlu simülasyon)."""
import numpy as np
import pytest

from app.services.vital_signs import VitalSignsService

FPS = 10
service = VitalSignsService()


def _detected(values):
    return service.analyze_temperature_series(list(values), fps=FPS)["status"] == "periodic_signal_detected"


def _noise(rng, kind, n):
    white = rng.normal(0, 1, n)
    if kind == "white":
        return white
    if kind == "walk":
        return np.cumsum(white)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(freqs), n)


@pytest.mark.parametrize("duration_s", [20, 60])
@pytest.mark.parametrize("kind", ["white", "walk", "pink"])
def test_pure_noise_rarely_reported_as_breathing(kind, duration_s):
    rng = np.random.default_rng(11)
    trials = 150
    hits = sum(_detected(_noise(rng, kind, duration_s * FPS)) for _ in range(trials))
    assert hits / trials <= 0.06, f"{kind} {duration_s}s yanlış olumlu oranı {hits / trials:.3f}"


@pytest.mark.parametrize("duration_s", [20, 60])
def test_clear_respiratory_component_is_detected(duration_s):
    rng = np.random.default_rng(3)
    n = duration_s * FPS
    t = np.arange(n) / FPS
    hits = sum(
        _detected(2.0 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28)) + rng.normal(0, 1, n))
        for _ in range(60)
    )
    assert hits / 60 >= 0.9


def test_slow_drift_with_no_oscillation_is_not_breathing():
    t = np.arange(300) / FPS
    drift = 0.5 * t + 0.02 * t ** 2
    assert not _detected(drift + np.random.default_rng(0).normal(0, 0.05, t.size))
