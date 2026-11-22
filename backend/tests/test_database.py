from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import AnalysisRecord


def test_analysis_record_can_be_persisted_and_read():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        item = AnalysisRecord(
            analysis_type="image",
            original_filename="thermal.jpg",
            created_at=datetime.now(timezone.utc),
            status="completed",
            candidate_count=3,
            summary={"candidate_region_count": 3},
            result_data={"method": "classical-threshold-contours"},
        )
        session.add(item)
        session.commit()
        session.refresh(item)
        record_id = item.id

    with Session() as session:
        saved = session.get(AnalysisRecord, record_id)
        assert saved is not None
        assert saved.original_filename == "thermal.jpg"
        assert saved.summary["candidate_region_count"] == 3
        assert saved.result_data["method"] == "classical-threshold-contours"
    engine.dispose()
