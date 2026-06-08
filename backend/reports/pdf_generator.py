"""
Phase M — Automated Clinical PDF Report Generator
Generates professional, publication-quality clinical reports using ReportLab.
"""
from __future__ import annotations
import base64
import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether,
    )
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(
    patient_info:   Dict[str, Any],
    prediction:     Dict[str, Any],
    quality:        Dict[str, Any],
    segmentation:   Dict[str, Any],
    localization:   Dict[str, Any],
    radiomics:      Dict[str, Any],
    neuroscore:     Dict[str, Any],
    risk:           Dict[str, Any],
    xai:            Dict[str, Any],
    uncertainty:    Dict[str, Any],
    original_image_path: str,
) -> bytes:
    """
    Build and return a full clinical PDF report as bytes.
    Falls back to a minimal text PDF if ReportLab is unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        return _minimal_text_pdf(patient_info, prediction, neuroscore, risk)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles  = getSampleStyleSheet()
    PRIMARY = HexColor("#00dbe7")
    DARK    = HexColor("#0f131f")
    MUTED   = HexColor("#849495")
    DANGER  = HexColor("#ef4444")
    WARNING = HexColor("#f59e0b")
    OK      = HexColor("#10b981")

    def style(name, **kw):
        s = styles[name].clone(f"custom_{name}_{id(kw)}")
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    H1 = style("Heading1", textColor=PRIMARY, fontSize=18, spaceAfter=6)
    H2 = style("Heading2", textColor=DARK,    fontSize=13, spaceAfter=4)
    H3 = style("Heading3", textColor=MUTED,   fontSize=11, spaceAfter=2, italic=True)
    BD = style("Normal",   fontSize=9,        leading=13)
    SM = style("Normal",   fontSize=8,        textColor=MUTED, leading=11)

    def sep():
        return HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6)

    def risk_color(level):
        return {"Low": OK, "Medium": WARNING, "High": DANGER}.get(level, MUTED)

    # ── Build story ───────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("NeuroIntel 3.0 — Clinical Decision Support System", H1))
    story.append(Paragraph("Automated Brain MRI Analysis Report", H3))
    story.append(Paragraph(
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"For Research Use Only — Not a Clinical Diagnosis", SM
    ))
    story.append(sep())

    # Patient Information
    story.append(Paragraph("Patient Information", H2))
    pdata = [
        ["Patient ID",        patient_info.get("patient_id", "N/A"),
         "Age",               str(patient_info.get("age", "N/A"))],
        ["Gender",            patient_info.get("gender", "N/A"),
         "Scan Date",         patient_info.get("scan_date", datetime.now().strftime("%Y-%m-%d"))],
        ["Referring Physician", patient_info.get("physician", "N/A"),
         "Institution",       patient_info.get("institution", "NeuroIntel Research Centre")],
        ["Symptoms",          patient_info.get("symptoms", "N/A"),
         "Family History",    "Yes" if patient_info.get("family_history") else "No"],
    ]
    pt = Table(pdata, colWidths=[3.5*cm, 5*cm, 3*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), HexColor("#f8fafc")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",   (2,0), (2,-1), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.25, MUTED),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [HexColor("#f8fafc"), colors.white]),
        ("PADDING",    (0,0), (-1,-1), 4),
    ]))
    story.append(pt); story.append(Spacer(1, 8))

    # Diagnosis Summary
    story.append(sep())
    story.append(Paragraph("AI Diagnosis Summary", H2))
    tumor_type = prediction.get("tumor_type", "N/A")
    conf       = prediction.get("confidence", 0)
    unc        = uncertainty.get("uncertainty_pct", 0)
    ns_score   = neuroscore.get("neuroscore", 0)
    ns_level   = neuroscore.get("level", "N/A")
    ns_col     = risk_color(ns_level.split()[0] if " " in ns_level else ns_level)

    diag_data = [
        ["Parameter",          "Value",                 "Interpretation"],
        ["Tumor Type",         tumor_type,              _tumor_note(tumor_type)],
        ["Model Confidence",   f"{conf:.1f}%",          "Monte Carlo Dropout estimate"],
        ["Uncertainty (±)",    f"±{unc:.1f}%",          "Epistemic uncertainty"],
        ["Region",             localization.get("region_label", "N/A"), "Approximate anatomical region"],
        ["Tumor Area",         f"{segmentation.get('area_pct', 0):.1f}%", "% of total image"],
        ["Volume (estimate)",  f"{segmentation.get('volume_mm3', 0):.0f} mm³", "1 mm isotropic assumption"],
        ["MRI Quality Score",  f"{quality.get('score', 0):.0f}/100", quality.get("status", "N/A").title()],
        ["NeuroScore",         f"{ns_score:.1f}/100",   ns_level],
    ]
    dt = Table(diag_data, colWidths=[4.5*cm, 4*cm, 8*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  DARK),
        ("TEXTCOLOR",     (0,0), (-1,0),  PRIMARY),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.25, MUTED),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, HexColor("#f8fafc")]),
        ("PADDING",       (0,0), (-1,-1), 5),
        # Colour NeuroScore row
        ("BACKGROUND",    (0,-1), (-1,-1), HexColor("#fff7ed")),
        ("TEXTCOLOR",     (1,-1), (1,-1),  ns_col),
        ("FONTNAME",      (1,-1), (1,-1),  "Helvetica-Bold"),
    ]))
    story.append(dt); story.append(Spacer(1, 8))

    # Images
    story.append(sep())
    story.append(Paragraph("Visual Analysis", H2))
    img_row = []
    _img_labels = []

    def b64_to_rl_image(b64_str: str, label: str, w=4.5*cm, h=4.5*cm):
        if not b64_str:
            return None, None
        try:
            raw = base64.b64decode(b64_str)
            buf_img = io.BytesIO(raw)
            return RLImage(buf_img, width=w, height=h), label
        except Exception:
            return None, None

    image_pairs = [
        (xai.get("original_b64",  ""), "Original MRI"),
        (xai.get("gradcam_b64",   ""), "Grad-CAM"),
        (xai.get("overlay_b64",   ""), "Grad-CAM Overlay"),
        (segmentation.get("overlay_b64", ""), "Segmentation"),
    ]
    row_imgs, row_lbls = [], []
    for b64, lbl in image_pairs:
        img, label = b64_to_rl_image(b64, lbl)
        if img:
            row_imgs.append(img)
            row_lbls.append(label)

    if row_imgs:
        # Max 4 per row
        chunks = [row_imgs[i:i+4] for i in range(0, len(row_imgs), 4)]
        lbl_chunks = [row_lbls[i:i+4] for i in range(0, len(row_lbls), 4)]
        for imgs, lbls in zip(chunks, lbl_chunks):
            img_table = Table(
                [imgs, [Paragraph(l, SM) for l in lbls]],
                colWidths=[4.5*cm] * len(imgs)
            )
            img_table.setStyle(TableStyle([
                ("ALIGN",   (0,0), (-1,-1), "CENTER"),
                ("PADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 6))

    # Radiomics
    story.append(sep())
    story.append(Paragraph("Radiomics Feature Summary", H2))
    feats = radiomics.get("features", {})
    if feats:
        rad_data = [["Feature", "Value", "Feature", "Value"]]
        feat_items = list(feats.items())
        for i in range(0, len(feat_items), 2):
            k1, v1 = feat_items[i]
            k2, v2 = feat_items[i+1] if i+1 < len(feat_items) else ("", "")
            rad_data.append([
                k1.replace("_", " "), f"{v1:.4f}" if isinstance(v1, float) else str(v1),
                k2.replace("_", " "), f"{v2:.4f}" if isinstance(v2, float) else str(v2),
            ])
        rt = Table(rad_data, colWidths=[5*cm, 3*cm, 5*cm, 3*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), PRIMARY),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7),
            ("GRID",          (0,0), (-1,-1), 0.25, MUTED),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, HexColor("#f8fafc")]),
            ("PADDING",       (0,0), (-1,-1), 3),
        ]))
        story.append(rt); story.append(Spacer(1, 8))

    # NeuroScore breakdown
    story.append(sep())
    story.append(Paragraph("NeuroScore Dimensional Breakdown", H2))
    dims = neuroscore.get("dimensions", {})
    wts  = neuroscore.get("weights", {})
    if dims:
        ns_data = [["Dimension", "Score", "Weight", "Contribution"]]
        dim_labels = {
            "tumor_severity":    "Tumor Type Severity",
            "model_confidence":  "Model Confidence",
            "tumor_size":        "Tumor Size",
            "mri_quality":       "MRI Quality",
            "radiomics_complexity": "Radiomics Complexity",
            "clinical_factors":  "Clinical Risk Factors",
        }
        for dk, dlbl in dim_labels.items():
            score = dims.get(dk, 0)
            wt    = wts.get(dk.replace("_complexity","").replace("_severity","_type")
                            .replace("_confidence","_confidence")
                            .replace("_factors","_risk"), 0)
            contribution = round(score * wt, 1)
            ns_data.append([dlbl, f"{score:.1f}", f"{int(wt*100)}%", f"{contribution:.1f}"])
        ns_data.append(["NEUROSCORE TOTAL", f"{ns_score:.1f} / 100", "", ns_level])
        nt = Table(ns_data, colWidths=[6*cm, 3*cm, 2.5*cm, 5*cm])
        nt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), PRIMARY),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,-1), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("GRID",          (0,0), (-1,-1), 0.25, MUTED),
            ("BACKGROUND",    (0,-1), (-1,-1), HexColor("#fff7ed")),
            ("TEXTCOLOR",     (1,-1), (1,-1),  ns_col),
            ("ROWBACKGROUNDS",(0,1), (-1,-2), [colors.white, HexColor("#f8fafc")]),
            ("PADDING",       (0,0), (-1,-1), 4),
        ]))
        story.append(nt); story.append(Spacer(1, 8))

    # Clinical Notes
    story.append(sep())
    story.append(Paragraph("AI Clinical Notes", H2))
    story.append(Paragraph(neuroscore.get("interpretation", ""), BD))
    story.append(Spacer(1, 4))
    story.append(Paragraph(risk.get("summary", ""), BD))
    if xai.get("explanation"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Grad-CAM Explanation:</b> " + xai["explanation"], BD))

    # Quality warnings
    if quality.get("warnings"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>MRI Quality Warnings:</b>", BD))
        for w in quality["warnings"]:
            story.append(Paragraph(f"• {w}", BD))

    # Disclaimer
    story.append(Spacer(1, 12))
    story.append(sep())
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for research purposes only. "
        "It is NOT a substitute for professional medical diagnosis, advice, or treatment. "
        "Always consult a qualified medical professional for clinical decisions.",
        SM
    ))
    story.append(Paragraph("© 2026 NeuroIntel Medical Systems | NeuroIntel 3.0", SM))

    doc.build(story)
    return buf.getvalue()


def _tumor_note(tumor_type: str) -> str:
    notes = {
        "glioma":     "Malignant; requires urgent specialist review",
        "meningioma": "Usually benign; may be asymptomatic",
        "pituitary":  "Typically benign adenoma; endocrine evaluation needed",
        "no tumor":   "No neoplastic findings detected",
    }
    for k, v in notes.items():
        if k in tumor_type.lower():
            return v
    return "—"


def _minimal_text_pdf(patient_info, prediction, neuroscore, risk) -> bytes:
    """Fallback plain-text-based PDF using basic bytes."""
    content = f"""NeuroIntel 3.0 — Clinical Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PATIENT: {patient_info.get('patient_id', 'N/A')}
DIAGNOSIS: {prediction.get('tumor_type', 'N/A')}
CONFIDENCE: {prediction.get('confidence', 0):.1f}%
NEUROSCORE: {neuroscore.get('neuroscore', 0):.1f}/100 ({neuroscore.get('level', 'N/A')})
RISK: {risk.get('risk_level', 'N/A')} ({risk.get('risk_score', 0):.0f}/100)

{neuroscore.get('interpretation', '')}

DISCLAIMER: For research use only.
"""
    return content.encode("utf-8")
