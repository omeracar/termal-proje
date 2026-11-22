from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from app.database import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(20), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    status = Column(String(20), nullable=False, default="completed")
    candidate_count = Column(Integer, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    summary = Column(JSON, nullable=False, default=dict)
    result_data = Column(JSON, nullable=False, default=dict)
    notes = Column(Text, nullable=True)
