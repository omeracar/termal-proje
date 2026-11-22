from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRecord
from app.services.report_service import build_analysis_pdf

router = APIRouter(prefix="/api/v1", tags=["Analysis History"])


def _record_json(record: AnalysisRecord):
    return {
        "id": record.id,
        "analysis_type": record.analysis_type,
        "original_filename": record.original_filename,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "status": record.status,
        "candidate_count": record.candidate_count,
        "duration_seconds": record.duration_seconds,
        "summary": record.summary or {},
    }


@router.get("/analyses")
def list_analyses(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc(), AnalysisRecord.id.desc())
    total = query.count()
    records = query.offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset,
            "items": [_record_json(item) for item in records]}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis record not found")
    return {**_record_json(record), "result_data": record.result_data or {}, "notes": record.notes}


@router.get("/analyses/{analysis_id}/report.pdf")
def download_report(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis record not found")
    pdf = build_analysis_pdf(record)
    filename = f"thermal-analysis-{record.id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
