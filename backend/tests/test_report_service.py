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
