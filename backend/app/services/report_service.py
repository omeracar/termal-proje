"""Kayıtlı analizler için indirilebilir PDF raporları üretir."""
import json
import os
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import unicodedata

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


def _load_eval_metrics():
    """data/eval_metrics.json (evaluate_detector.py çıktısı) varsa okur; yoksa None döner."""
    default = Path(__file__).resolve().parents[3] / "data" / "eval_metrics.json"
    path = Path(os.getenv("EVAL_METRICS_PATH", str(default)))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data.get("summary"), dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def _safe(value):
    if value is None:
        value = "-"
    if isinstance(value, (dict, list)):
        value = str(value)
    text = str(value)
    if "ThermalUnicode" not in pdfmetrics.getRegisteredFontNames():
        replacements = str.maketrans({
            "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S",
            "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
            "—": "-", "–": "-", "’": "'", "“": '"', "”": '"',
        })
        text = unicodedata.normalize("NFKD", text.translate(replacements)).encode("ascii", "ignore").decode("ascii")
    return escape(text)


def build_analysis_pdf(record) -> bytes:
    font_name = _register_unicode_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=f"Termal analiz raporu #{record.id}",
        author="Termal Kamera Analiz Prototipi",
    )
    styles = getSampleStyleSheet()
    for style_name in ("Normal", "BodyText", "Title", "Heading2"):
        styles[style_name].fontName = font_name
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21, fontName=font_name))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#555555"), fontName=font_name))
    story = [
        Paragraph("TERMAL KAMERA ANALİZ RAPORU", styles["ReportTitle"]),
        Spacer(1, 5 * mm),
        Paragraph("Araştırma prototipi - karar destek çıktısıdır; tıbbi tanı ya da yaşam/ölüm kanıtı değildir.", styles["SmallNote"]),
        Spacer(1, 5 * mm),
    ]

    created = record.created_at.isoformat() if record.created_at else "—"
    metadata = [
        ["Analiz no", str(record.id)],
        ["Analiz türü", record.analysis_type.upper()],
        ["Kaynak dosya", record.original_filename],
        ["Oluşturulma zamanı", created],
        ["Status", record.status],
        ["Aday bölge / iz sayısı", str(record.candidate_count if record.candidate_count is not None else "—")],
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
    story.extend([table, Spacer(1, 6 * mm), Paragraph("Analiz özeti", styles["Heading2"])])

    summary = record.summary or {}
    rows = [[Paragraph("<b>Ölçüt</b>", styles["BodyText"]), Paragraph("<b>Değer</b>", styles["BodyText"])]]
    for key, value in summary.items():
        rows.append([Paragraph(_safe(key.replace("_", " ").title()), styles["BodyText"]),
                     Paragraph(_safe(value), styles["BodyText"])])
    if len(rows) == 1:
        rows.append([Paragraph("Özet ölçüt yok", styles["BodyText"]), Paragraph("—", styles["BodyText"])])
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
    story.extend([summary_table, Spacer(1, 6 * mm), Paragraph("Teknik ayrıntılar", styles["Heading2"])])

    details = record.result_data or {}
    for section, payload in details.items():
        story.append(Paragraph(f"<b>{_safe(section.replace('_', ' ').title())}</b>", styles["BodyText"]))
        story.append(Paragraph(_safe(payload)[:5000], styles["SmallNote"]))
        story.append(Spacer(1, 2 * mm))

    metrics = _load_eval_metrics()
    if metrics:
        m = metrics["summary"]
        story.extend([Spacer(1, 4 * mm), Paragraph("Dedektör doğrulama ölçümleri", styles["Heading2"])])
        metric_rows = [
            ["Etiketli görüntü sayısı", str(m.get("images", "—"))],
            ["Doğru tespit (TP)", str(m.get("tp", "—"))],
            ["Yanlış tespit (FP)", str(m.get("fp", "—"))],
            ["Tespit edilemeyen kişi (FN)", str(m.get("fn", "—"))],
            ["Precision", str(m.get("precision", "—"))],
            ["Recall", str(m.get("recall", "—"))],
            ["F1-Score", str(m.get("f1", "—"))],
            ["Görüntü başına ortalama süre (s)", str(m.get("mean_seconds_per_image", "—"))],
            ["Eşleşme eşiği (IoU)", str(m.get("iou_threshold", "—"))],
        ]
        metric_table = Table([[Paragraph(_safe(a), styles["BodyText"]), Paragraph(_safe(b), styles["BodyText"])]
                              for a, b in metric_rows], colWidths=[85 * mm, 81 * mm])
        metric_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D8")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([
            metric_table,
            Paragraph("Bu ölçümler elle etiketlenmiş küçük bir gerçek görüntü kümesine aittir; bu kayıttaki "
                      "görüntünün başarımını değil, dedektörün genel durumunu gösterir ve genelleştirilemez.",
                      styles["SmallNote"]),
        ])

    story.extend([
        Spacer(1, 4 * mm),
        Paragraph(
            "ÖNEMLİ: Aday bölgeler doğrulanmış kişi değildir. Piksel yoğunluğu (DN) fiziksel sıcaklık (°C) değildir. "
            "Periyodik sinyal bileşenleri nefes almanın veya yaşamın kanıtı değildir; bulunamamaları da ölümün kanıtı değildir. "
            "Doğrulanmamış bu prototip, kurtarma, triyaj ya da yaşam/ölüm kararlarında tek dayanak olarak kullanılmamalıdır.",
            styles["SmallNote"],
        ),
    ])
    doc.build(story)
    return buffer.getvalue()
