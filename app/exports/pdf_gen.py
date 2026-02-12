import logging
import os
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)

logger = logging.getLogger(__name__)

styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "CustomTitle", parent=styles["Title"], fontSize=20, spaceAfter=20
)
HEADING_STYLE = ParagraphStyle(
    "CustomHeading", parent=styles["Heading2"], fontSize=14, spaceBefore=16, spaceAfter=8
)
BODY_STYLE = ParagraphStyle(
    "CustomBody", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6
)
SMALL_STYLE = ParagraphStyle(
    "Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.grey
)


def generate_pdf(upload_id: str, results: dict, output_dir: str) -> str:
    """Generate a PDF report from analysis results."""
    logger.info(f"Generating PDF for {upload_id}")

    filepath = os.path.join(output_dir, f"report_{upload_id}.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)

    elements = []

    # title page
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("DataWhisperer Analysis Report", TITLE_STYLE))
    elements.append(Spacer(1, 0.5 * inch))

    metadata = results.get("metadata", {})
    if metadata:
        elements.append(Paragraph(f"Dataset: {metadata.get('filename', 'Unknown')}", BODY_STYLE))
        elements.append(Paragraph(
            f"{metadata.get('rows', '?')} rows x {metadata.get('columns', '?')} columns | "
            f"{metadata.get('file_size_mb', '?')} MB",
            BODY_STYLE
        ))

    report = results.get("report", {})
    quality = report.get("quality_score", "N/A")
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Quality Score: {quality}/100", HEADING_STYLE))
    elements.append(PageBreak())

    # data profile section
    elements.append(Paragraph("Data Profile", HEADING_STYLE))
    profile = results.get("profile", {})
    if profile:
        elements.extend(_build_profile_table(profile))
    elements.append(Spacer(1, 0.3 * inch))

    # quality findings
    detective = results.get("detective_results", {})
    findings = detective.get("findings", [])
    if findings:
        elements.append(Paragraph("Data Quality Findings", HEADING_STYLE))
        elements.extend(_build_findings_table(findings))
        elements.append(Spacer(1, 0.3 * inch))

    # embed visualizations as images
    visualizations = results.get("visualizations", [])
    if visualizations:
        elements.append(PageBreak())
        elements.append(Paragraph("Visualizations", HEADING_STYLE))
        for viz in visualizations:
            fig = viz.get("figure")
            if fig is None:
                continue
            try:
                img_bytes = fig.to_image(format="png", width=700, height=400)
                img_buffer = BytesIO(img_bytes)
                img = Image(img_buffer, width=6 * inch, height=3.4 * inch)
                elements.append(Paragraph(viz.get("title", ""), BODY_STYLE))
                elements.append(img)
                elements.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                logger.debug(f"Could not embed chart in PDF: {e}")
                elements.append(Paragraph(f"[Chart: {viz.get('title', 'N/A')}]", SMALL_STYLE))

    # narrative report
    report_text = report.get("report_text", "")
    if report_text:
        elements.append(PageBreak())
        elements.append(Paragraph("Analysis Report", HEADING_STYLE))
        for para in report_text.split("\n\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para.replace("\n", "<br/>"), BODY_STYLE))
                elements.append(Spacer(1, 0.1 * inch))

    try:
        doc.build(elements)
        logger.info(f"PDF saved to {filepath}")
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise

    return filepath


def _build_profile_table(profile: dict) -> list:
    """Build a table showing column profiles."""
    elements = []

    summary_data = [
        ["Metric", "Value"],
        ["Rows", str(profile.get("shape", {}).get("rows", ""))],
        ["Columns", str(profile.get("shape", {}).get("columns", ""))],
        ["Completeness", f"{profile.get('completeness_pct', '')}%"],
        ["Duplicate Rows", str(profile.get("duplicate_rows", ""))],
    ]
    t = Table(summary_data, colWidths=[2.5 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2 * inch))

    # column details
    columns = profile.get("columns", {})
    if columns:
        col_data = [["Column", "Type", "Null %", "Unique"]]
        for name, info in columns.items():
            col_data.append([
                name[:30],
                info.get("detected_type", ""),
                f"{info.get('null_pct', 0)}%",
                str(info.get("n_unique", "")),
            ])

        t2 = Table(col_data, colWidths=[2 * inch, 1.2 * inch, 1 * inch, 1 * inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t2)

    return elements


def _build_findings_table(findings: list) -> list:
    elements = []
    data = [["Severity", "Issue", "Recommendation"]]

    severity_colors = {
        "high": colors.HexColor("#FF4444"),
        "medium": colors.HexColor("#FFAA00"),
        "low": colors.HexColor("#44AA44"),
    }

    for f in findings:
        data.append([
            f.get("severity", "").upper(),
            f.get("title", "")[:60],
            f.get("recommendation", "")[:80],
        ])

    t = Table(data, colWidths=[0.8 * inch, 2.5 * inch, 3 * inch])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]

    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "low")
        color = severity_colors.get(sev, colors.grey)
        style_cmds.append(("TEXTCOLOR", (0, i), (0, i), color))

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    return elements
