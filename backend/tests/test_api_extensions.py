"""Çevre karşılaştırması, dağılım, ısı haritası, video önizleme ve sonuç ifadeleri testleri."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.services.temperature_analysis import TemperatureAnalysisService
from app.services.visualization import build_heatmap
from app.services.vital_signs import (
    FINDING_COMPATIBLE, FINDING_INSUFFICIENT, FINDING_UNDETERMINED, VitalSignsService,
)

REGION = {"x": 20, "y": 20, "w": 20, "h": 40}


def _scene():
    image = np.full((100, 80), 60, dtype=np.uint8)
    image[20:60, 20:40] = 200
    return image


def test_surroundings_comparison_uses_relative_dn_label():
    result = TemperatureAnalysisService().compare_with_surroundings(_scene(), REGION)
    assert result["available"] and result["unit"] == "DN"
    assert result["candidate_mean"] == pytest.approx(200)
    assert result["background_mean"] == pytest.approx(60)
    assert result["difference"] == pytest.approx(140)
    assert "görece" in result["note"].lower()


def test_surroundings_comparison_celsius_when_map_supplied():
    tmap = np.full((100, 80), 20.0, dtype=np.float32)
    tmap[20:60, 20:40] = 34.0
    result = TemperatureAnalysisService().compare_with_surroundings(_scene(), REGION, tmap)
    assert result["unit"] == "degC"
    assert result["difference"] == pytest.approx(14.0)
    assert "note" not in result


def test_distribution_histogram_counts_all_pixels():
    dist = TemperatureAnalysisService().region_distribution(_scene(), REGION)
    assert dist["unit"] == "DN"
    assert sum(dist["counts"]) == 20 * 40
    assert len(dist["bin_edges"]) == len(dist["counts"]) + 1


def test_heatmap_has_colorbar_and_marks_candidates():
    heat = build_heatmap(_scene(), [REGION])
    assert heat.ndim == 3 and heat.shape[0] == 100 and heat.shape[1] > 80
    assert tuple(heat[20, 25]) == (255, 255, 255)  # kutu kenarı beyaz


def test_findings_match_specification_sentences():
    service = VitalSignsService()
    assert service.analyze_temperature_series([1.0] * 20, fps=10)["finding"] == FINDING_INSUFFICIENT
    assert service.analyze_temperature_series([10.0] * 300, fps=10)["finding"] == FINDING_UNDETERMINED
    t = np.arange(600) / 10.0
    rng = np.random.default_rng(0)
    breathing = 5 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.2, t.size)
    result = service.analyze_temperature_series(breathing.tolist(), fps=10)
    assert result["status"] == "periodic_signal_detected"
    assert result["finding"] == FINDING_COMPATIBLE


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def _person_image():
    image = np.full((240, 320), 40, dtype=np.uint8)
    cv2.ellipse(image, (160, 110), (22, 28), 0, 0, 360, 210, -1)
    cv2.rectangle(image, (138, 130), (182, 220), 190, -1)
    return cv2.GaussianBlur(image, (5, 5), 0)


def test_image_api_returns_comparison_distribution_heatmap_and_finding(client):
    http, _ = client
    ok, buf = cv2.imencode(".png", _person_image())
    response = http.post("/api/v1/analyze", files={"file": ("t.png", buf.tobytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["heatmap_image"].startswith("data:image/jpeg;base64,")
    assert "görece" in body["heatmap_note"]
    assert body["vital_signs"]["finding"] == FINDING_INSUFFICIENT
    for item in body["temperature_analysis"]:
        assert item["background_comparison"]["unit"] == "DN"
        assert item["distribution"]["unit"] == "DN"


def test_video_api_returns_previews_but_does_not_store_them(client, tmp_path):
    http, Session = client
    path = str(tmp_path / "v.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240), False)
    base = _person_image()
    for i in range(120):  # 12 sn
        writer.write(np.roll(base, i % 3, axis=1))
    writer.release()
    with open(path, "rb") as fh:
        response = http.post("/api/v1/analyze-video", files={"file": ("v.avi", fh.read(), "video/x-msvideo")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert 1 <= len(body["preview_frames"]) <= 3
    raw = base64.b64decode(body["preview_frames"][0]["image"].split(",", 1)[1])
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None
    for track in body["tracks"]:
        assert track["vital_signs"]["finding"] in (FINDING_COMPATIBLE, FINDING_UNDETERMINED, FINDING_INSUFFICIENT)
    from app.models import AnalysisRecord
    with Session() as db:
        record = db.query(AnalysisRecord).one()
        assert "preview_frames" not in (record.result_data or {})
