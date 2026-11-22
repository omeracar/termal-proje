"""Create downloadable PDF reports for stored analysis records."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def _register_unicode_font():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            if "ThermalUnicode" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("ThermalUnicode", regular))
                pdfmetrics.registerFont(TTFont("ThermalUnicode-Bold", bold))
                pdfmetrics.registerFontFamily("ThermalUnicode", normal="ThermalUnicode", bold="ThermalUnicode-Bold")
            return "ThermalUnicode"
    return "Helvetica"


def _safe(value):
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        value = str(value)
    return escape(str(value))


def build_analysis_pdf(record) -> bytes:
    font_name = _register_unicode_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=f"Thermal analysis report #{record.id}",
        author="Thermal Camera Analysis Prototype",
    )
    styles = getSampleStyleSheet()
    for style_name in ("Normal", "BodyText", "Title", "Heading2"):
        styles[style_name].fontName = font_name
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21, fontName=font_name))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#555555"), fontName=font_name))
    story = [
        Paragraph("THERMAL CAMERA ANALYSIS REPORT", styles["ReportTitle"]),
        Spacer(1, 5 * mm),
        Paragraph("Research prototype — decision-support output, not a medical diagnosis or proof of life/death.", styles["SmallNote"]),
        Spacer(1, 5 * mm),
    ]

    created = record.created_at.isoformat() if record.created_at else "—"
    metadata = [
        ["Analysis ID", str(record.id)],
        ["Analysis type", record.analysis_type.upper()],
        ["Source file", record.original_filename],
        ["Created at", created],
        ["Status", record.status],
        ["Candidate regions / tracks", str(record.candidate_count if record.candidate_count is not None else "—")],
    ]
    table = Table([[Paragraph(_safe(k), styles["BodyText"]), Paragraph(_safe(v), styles["BodyText"])] for k, v in metadata], colWidths=[48 * mm, 118 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D0D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 6 * mm), Paragraph("Analysis summary", styles["Heading2"])])

    summary = record.summary or {}
    rows = [[Paragraph("<b>Metric</b>", styles["BodyText"]), Paragraph("<b>Value</b>", styles["BodyText"])]]
    for key, value in summary.items():
        rows.append([Paragraph(_safe(key.replace("_", " ").title()), styles["BodyText"]),
                     Paragraph(_safe(value), styles["BodyText"])])
    if len(rows) == 1:
        rows.append([Paragraph("No summary metrics", styles["BodyText"]), Paragraph("—", styles["BodyText"])])
    summary_table = Table(rows, colWidths=[65 * mm, 101 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6EF")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([summary_table, Spacer(1, 6 * mm), Paragraph("Technical details", styles["Heading2"])])

    details = record.result_data or {}
    for section, payload in details.items():
        story.append(Paragraph(f"<b>{_safe(section.replace('_', ' ').title())}</b>", styles["BodyText"]))
        story.append(Paragraph(_safe(payload)[:5000], styles["SmallNote"]))
        story.append(Spacer(1, 2 * mm))

    story.extend([
        Spacer(1, 4 * mm),
        Paragraph(
            "IMPORTANT: Candidate regions are not confirmed people. Pixel intensity (DN) is not physical temperature (°C). "
            "Periodic signal components are not proof of breathing or life, and failure to detect them is not proof of death. "
            "This unvalidated prototype must not be used as the sole basis for rescue, triage, or life/death decisions.",
            styles["SmallNote"],
        ),
    ])
    doc.build(story)
    return buffer.getvalue()
