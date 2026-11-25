from datetime import datetime, timezone
from types import SimpleNamespace

from app.services.report_service import build_analysis_pdf


def test_analysis_pdf_is_generated():
    record = SimpleNamespace(
        id=17,
        analysis_type="image",
        original_filename="termal_goruntu.jpg",
        created_at=datetime(2026, 11, 22, tzinfo=timezone.utc),
        status="completed",
        candidate_count=2,
        summary={"candidate_region_count": 2, "temperature_unit": "DN"},
        result_data={
            "method": "classical-threshold-contours",
            "temperature_warning": "Piksel yoğunluğu gerçek sıcaklık değildir.",
        },
    )
    pdf = build_analysis_pdf(record)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500


def test_pdf_includes_detector_metrics_when_file_exists(tmp_path, monkeypatch):
    import json
    from app.services import report_service
    metrics = tmp_path / "m.json"
    metrics.write_text(json.dumps({"summary": {"images": 10, "tp": 14, "fp": 31, "fn": 41,
                                               "precision": 0.311, "recall": 0.255, "f1": 0.28,
                                               "mean_seconds_per_image": 0.107, "iou_threshold": 0.5}}))
    monkeypatch.setenv("EVAL_METRICS_PATH", str(metrics))
    assert report_service._load_eval_metrics()["summary"]["f1"] == 0.28
    monkeypatch.setenv("EVAL_METRICS_PATH", str(tmp_path / "yok.json"))
    assert report_service._load_eval_metrics() is None
