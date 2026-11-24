import numpy as np
import pytest

from app.services.image_processing import ImageProcessingService, to_gray
from app.services.temperature_analysis import TemperatureAnalysisService
from app.services.vital_signs import VitalSignsService


def test_to_gray_preserves_uint8_grayscale():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert np.array_equal(to_gray(image), image)


def test_to_gray_scales_16bit_input():
    image = np.zeros((20, 20), dtype=np.uint16)
    image[5:15, 5:15] = 30000
    gray = to_gray(image)
    assert gray.dtype == np.uint8
    assert gray.max() == 255


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        to_gray(np.array([], dtype=np.uint8))


def test_temperature_analysis_does_not_claim_celsius_for_pixel_data():
    image = np.full((20, 20), 180, dtype=np.uint8)
    result = TemperatureAnalysisService().analyze_region(image, {"x": 2, "y": 2, "w": 8, "h": 8})
    assert result["measurement_type"] == "raw_pixel_intensity"
    assert result["unit"] == "DN"
    assert result["calibration_provided"] is False


def test_temperature_map_reports_celsius_only_when_explicitly_supplied():
    image = np.zeros((20, 20), dtype=np.uint8)
    temperature_map = np.full((20, 20), 21.5, dtype=np.float32)
    result = TemperatureAnalysisService().analyze_region(
        image, {"x": 2, "y": 2, "w": 8, "h": 8}, temperature_map_c=temperature_map
    )
    assert result["measurement_type"] == "radiometric_temperature"
    assert result["unit"] == "degC"
    assert result["mean"] == pytest.approx(21.5)


def test_temperature_map_shape_mismatch_is_rejected():
    image = np.zeros((20, 20), dtype=np.uint8)
    temperature_map = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        TemperatureAnalysisService().analyze_region(
            image, {"x": 2, "y": 2, "w": 8, "h": 8}, temperature_map_c=temperature_map
        )


def test_vital_analysis_requires_minimum_duration():
    result = VitalSignsService().analyze_temperature_series([1.0] * 20, fps=10)
    assert result["status"] == "insufficient_data"


def test_constant_signal_is_not_reported_as_breathing():
    result = VitalSignsService().analyze_temperature_series([10.0] * 300, fps=10)
    assert result["status"] == "not_detected"


def test_candidate_detector_returns_classical_method():
    image = np.zeros((120, 100), dtype=np.uint8)
    image[20:100, 35:65] = 220
    candidates, method = ImageProcessingService().detect_people(image)
    assert method == "classical-multilevel-contours"
    assert isinstance(candidates, list)
    for item in candidates:
        assert item["candidate_type"] == "thermal_geometric_candidate"
