from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_JSON = OUTPUT_DIR / "spectral_zone_report.json"
FALLBACK_REPORT_JSON = OUTPUT_DIR / "report.json"
PDF_OUTPUT = OUTPUT_DIR / "terrain_change_report.pdf"

# Import lazily so the generator remains usable in isolation.
def _get_zone_evidence(region_id: int):
    from backend.evidence import get_zone_evidence
    return get_zone_evidence(region_id)


def _load_report() -> dict[str, Any]:
    for path in (REPORT_JSON, FALLBACK_REPORT_JSON):
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "No TERRAIN analysis report found. Run /analyze-area first."
    )


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_area(value: Any) -> str:
    try:
        return f"{float(value):,.1f} m²"
    except (TypeError, ValueError):
        return "—"


def _safe_text(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _get_zones(report: dict[str, Any]) -> list[dict[str, Any]]:
    zones = report.get("zones", [])
    if isinstance(zones, list):
        return [z for z in zones if isinstance(z, dict)]
    return []


def _zone_properties(zone: dict[str, Any]) -> dict[str, Any]:
    # Support both GeoJSON-like properties and report-native zone records.
    props = zone.get("properties")
    if isinstance(props, dict):
        return props
    return zone


def _build_direction_counts(zones: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for zone in zones:
        props = _zone_properties(zone)
        direction = _safe_text(props.get("direction"), "uncertain")
        counts[direction] = counts.get(direction, 0) + 1
    return counts


def _build_severity_counts(zones: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for zone in zones:
        props = _zone_properties(zone)
        severity = _safe_text(props.get("severity"), "unknown")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _make_styles():
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "TerrainTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "TerrainSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#64748B"),
        ),
        "section": ParagraphStyle(
            "TerrainSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=7,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "TerrainBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
        ),
        "small": ParagraphStyle(
            "TerrainSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=10,
            textColor=colors.HexColor("#64748B"),
        ),
        "metric": ParagraphStyle(
            "TerrainMetric",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_LEFT,
        ),
        "metric_label": ParagraphStyle(
            "TerrainMetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#64748B"),
        ),
        "table": ParagraphStyle(
            "TerrainTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#334155"),
        ),
        "table_head": ParagraphStyle(
            "TerrainTableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#0F172A"),
        ),
        "center": ParagraphStyle(
            "TerrainCenter",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155"),
            alignment=TA_CENTER,
        ),
    }


def _footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(16 * mm, 12 * mm, width - 16 * mm, 12 * mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(colors.HexColor("#617789"))
    canvas.drawString(16 * mm, 7.5 * mm, "TERRAIN • AI-Driven Satellite Image Change Intelligence System")
    canvas.drawRightString(width - 16 * mm, 7.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _metric_table(items, styles):
    cells = []
    for label, value in items:
        cells.append(
            [
                Paragraph(label.upper(), styles["metric_label"]),
                Paragraph(_safe_text(value), styles["metric"]),
            ]
        )

    data = [cells[i:i + 2] for i in range(0, len(cells), 2)]
    flattened = []
    for row in data:
        row_cells = []
        for cell in row:
            row_cells.extend(cell)
        while len(row_cells) < 4:
            row_cells.extend(["", ""])
        flattened.append(row_cells)

    table = Table(flattened, colWidths=[30 * mm, 48 * mm, 30 * mm, 48 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _distribution_table(title, counts, styles):
    rows = [[Paragraph(title.upper(), styles["table_head"]), Paragraph("COUNT", styles["table_head"])]]
    for key, value in counts.items():
        rows.append([Paragraph(key.replace("_", " ").title(), styles["table"]), Paragraph(str(value), styles["table"])])
    table = Table(rows, colWidths=[75 * mm, 25 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _zone_table(zones, styles):
    rows = [[
        Paragraph("ZONE", styles["table_head"]),
        Paragraph("AREA", styles["table_head"]),
        Paragraph("DIRECTION", styles["table_head"]),
        Paragraph("SEVERITY", styles["table_head"]),
        Paragraph("CONFIDENCE", styles["table_head"]),
        Paragraph("NDVI Δ", styles["table_head"]),
    ]]

    for zone in zones:
        p = _zone_properties(zone)
        ndvi = p.get("ndvi_evidence") or {}
        rows.append([
            Paragraph(f"Zone {_safe_text(p.get('region_id'))}", styles["table"]),
            Paragraph(_fmt_area(p.get("area_m2", p.get("area"))), styles["table"]),
            Paragraph(_safe_text(p.get("direction"), "uncertain").replace("_", " ").title(), styles["table"]),
            Paragraph(_safe_text(p.get("severity"), "unknown").title(), styles["table"]),
            Paragraph(_fmt(p.get("confidence"), 3), styles["table"]),
            Paragraph(_fmt(ndvi.get("change"), 3), styles["table"]),
        ])

    table = Table(
        rows,
        colWidths=[20 * mm, 30 * mm, 39 * mm, 25 * mm, 28 * mm, 22 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _image_from_bytes(data: bytes, width: float, height: float | None = None):
    stream = io.BytesIO(data)
    image = Image(stream, width=width, height=height) if height else Image(stream, width=width)
    image.hAlign = "LEFT"
    return image


def generate_report_pdf(output_path: Path | None = None) -> Path:
    output_path = output_path or PDF_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = _load_report()
    styles = _make_styles()

    roi = report.get("roi") or {}
    detection = report.get("detection") or {}
    metadata = report.get("metadata") or {}
    analysis_request = metadata.get("analysis_request") or {}
    satellite = metadata.get("satellite_acquisition") or {}
    zones = _get_zones(report)

    zones_sorted = sorted(
        zones,
        key=lambda z: _num(_zone_properties(z).get("area_m2", _zone_properties(z).get("area")), 0),
        reverse=True,
    )

    changed_pixels = detection.get("changed_pixels", 0)
    roi_pixels = roi.get("roi_pixels", 0)
    changed_percentage = detection.get("changed_percentage", 0)
    zone_count = detection.get("grouped_regions", len(zones))
    raw_regions = detection.get("raw_regions", 0)
    threshold = detection.get("threshold", 0)

    before_date = analysis_request.get("before_date") or satellite.get("before", {}).get("datetime") or "—"
    after_date = analysis_request.get("after_date") or satellite.get("after", {}).get("datetime") or "—"

    lat = analysis_request.get("latitude", report.get("input", {}).get("latitude", "—"))
    lon = analysis_request.get("longitude", report.get("input", {}).get("longitude", "—"))
    radius = analysis_request.get("radius_m", report.get("input", {}).get("radius_m", "—"))

    direction_counts = _build_direction_counts(zones)
    severity_counts = _build_severity_counts(zones)

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title="TERRAIN Satellite Change Analysis Report",
        author="TERRAIN",
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
    )
    doc.addPageTemplates([PageTemplate(id="terrain", frames=frame, onPage=_footer)])

    story = []

    # Header
    story.append(Paragraph("TERRAIN", styles["title"]))
    story.append(Paragraph(
        "AI-Driven Satellite Image Change Intelligence System",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0284C7")))
    story.append(Spacer(1, 10))
    story.append(Paragraph("SATELLITE CHANGE ANALYSIS REPORT", styles["section"]))

    story.append(_metric_table([
        ("Latitude", _fmt(lat, 6)),
        ("Longitude", _fmt(lon, 6)),
        ("Radius", f"{_fmt(radius, 0)} m"),
        ("Before", _safe_text(before_date)),
        ("After", _safe_text(after_date)),
        ("Valid ROI", f"{roi_pixels:,} pixels"),
        ("Changed Pixels", f"{int(_num(changed_pixels)):,}"),
        ("Changed ROI", f"{_fmt(changed_percentage, 2)}%"),
    ], styles))

    story.append(Spacer(1, 10))
    story.append(Paragraph("EXECUTIVE SUMMARY", styles["section"]))
    story.append(Paragraph(
        f"TERRAIN detected {int(_num(zone_count))} grouped change zones "
        f"from {int(_num(changed_pixels)):,} changed pixels within a valid ROI "
        f"of {int(_num(roi_pixels)):,} pixels. The detected change covers "
        f"{_fmt(changed_percentage, 2)}% of the valid ROI. The numerical detector "
        f"used an adaptive threshold of {_fmt(threshold, 3)} after temporal "
        f"harmonization and spectral/NDVI evidence fusion.",
        styles["body"],
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("CHANGE DISTRIBUTION", styles["section"]))
    dist_table = Table(
        [
            [
                _distribution_table("Direction", direction_counts, styles),
                _distribution_table("Severity", severity_counts, styles),
            ]
        ],
        colWidths=[87 * mm, 87 * mm],
    )
    dist_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(dist_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("TOP CHANGE ZONES", styles["section"]))
    story.append(_zone_table(zones_sorted[:12], styles))

    if zones_sorted:
        major = _zone_properties(zones_sorted[0])
        major_id = int(_num(major.get("region_id"), 0))

        try:
            evidence = _get_zone_evidence(major_id)
        except Exception:
            evidence = None

        story.append(PageBreak())
        story.append(Paragraph(f"MAJOR ZONE EVIDENCE • ZONE {major_id}", styles["section"]))

        ndvi = major.get("ndvi_evidence") or {}
        change = major.get("change_evidence") or {}

        story.append(_metric_table([
            ("Area", _fmt_area(major.get("area_m2", major.get("area")))),
            ("Direction", _safe_text(major.get("direction"), "uncertain").replace("_", " ").title()),
            ("Severity", _safe_text(major.get("severity"), "unknown").title()),
            ("Detector Confidence", _fmt(major.get("confidence"), 3)),
            ("NDVI Before", _fmt(ndvi.get("before"), 3)),
            ("NDVI After", _fmt(ndvi.get("after"), 3)),
            ("NDVI Change", _fmt(ndvi.get("change"), 3)),
            ("Spectral Contrast", _fmt(change.get("mean_spectral_distance"), 3)),
        ], styles))

        if evidence:
            before = evidence.get("before_png")
            after = evidence.get("after_png")
            overlay = evidence.get("overlay_png")

            if before and after:
                image_table = Table(
                    [[
                        _image_from_bytes(before, 82 * mm, 72 * mm),
                        _image_from_bytes(after, 82 * mm, 72 * mm),
                    ], [
                        Paragraph("BEFORE", styles["center"]),
                        Paragraph("AFTER", styles["center"]),
                    ]],
                    colWidths=[87 * mm, 87 * mm],
                )
                image_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(image_table)
                story.append(Spacer(1, 8))

            if overlay:
                story.append(Paragraph("CHANGE OVERLAY", styles["section"]))
                story.append(_image_from_bytes(overlay, 110 * mm, 95 * mm))

    story.append(PageBreak())
    story.append(Paragraph("METHODOLOGY", styles["section"]))
    methodology = [
        "Sentinel-2 L2A imagery is acquired for the requested before and after periods.",
        "B02, B03, B04 and B08 are used as the multispectral evidence bands.",
        "Invalid/cloud-affected pixels are excluded using the valid-pixel mask.",
        "Temporal harmonization is applied before physical spectral comparison.",
        "NDVI is derived from B08 and B04 for temporal vegetation evidence.",
        "Spectral distance and absolute NDVI change are fused into a pixel-level change score.",
        "An adaptive threshold identifies candidate changed pixels.",
        "Connected-component extraction and spatial grouping produce geographic change zones.",
        "Zone metrics include area, NDVI change, spectral contrast, direction and evidence confidence.",
        "Qwen 2.5-VL is used only for semantic interpretation of already detected zones.",
    ]
    for item in methodology:
        story.append(Paragraph(f"• {item}", styles["body"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 8))
    story.append(Paragraph("LIMITATIONS", styles["section"]))
    limitations = [
        "Detector confidence is an evidence score, not classification accuracy.",
        "Semantic interpretation is not ground-truth verification.",
        "NDVI decline alone does not prove deforestation, construction or another specific cause.",
        "NDVI increase alone does not prove ecological recovery.",
        "Cloud, seasonal, illumination and acquisition differences can influence temporal evidence.",
        "Final interpretation should be reviewed against higher-resolution imagery or field evidence when decisions are consequential.",
    ]
    for item in limitations:
        story.append(Paragraph(f"• {item}", styles["body"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"TERRAIN report generated from the current verified analysis. "
        f"Raw candidate regions: {int(_num(raw_regions))}; grouped zones: {int(_num(zone_count))}.",
        styles["small"],
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    path = generate_report_pdf()
    print(f"TERRAIN PDF generated: {path}")
