"""
Phase H+I — Multimodal Risk Assessment & Novel NeuroScore System
Combines image features, radiomics, and patient clinical metadata.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List


# ─── Clinical Risk Factors ────────────────────────────────────────────────────
TUMOR_TYPE_BASE_RISK = {
    "glioma":     80,   # Highest baseline — high-grade potential
    "meningioma": 50,   # Moderate — usually benign but variable
    "pituitary":  40,   # Low-moderate — mostly benign adenomas
    "notumor":     0,   # No tumour → no base risk
}

SYMPTOM_RISK_MAP = {
    "severe_headache":       8,
    "seizures":             10,
    "vision_loss":           7,
    "cognitive_decline":    12,
    "motor_weakness":        9,
    "speech_difficulty":     8,
    "nausea_vomiting":       5,
    "hormonal_imbalance":    6,
    "none":                  0,
}


# ─── Phase H: Multimodal Risk Assessment ─────────────────────────────────────

def compute_risk(
    tumor_type:       str,
    confidence:       float,          # 0-100
    area_pct:         float,          # 0-100
    volume_mm3:       float,
    mri_quality:      float,          # 0-100
    radiomics_features: Dict[str, Any],
    patient_age:      int   = 0,
    patient_gender:   str   = "unknown",
    symptoms:         List[str] = None,
    family_history:   bool  = False,
    prior_tumor:      bool  = False,
) -> Dict[str, Any]:
    """
    Multimodal risk assessment combining image, radiomics, and clinical inputs.

    Returns risk_score (0-100), risk_level, component_scores, and summary.
    """
    if symptoms is None:
        symptoms = []

    tumor_key = tumor_type.lower().replace(" ", "").replace("tumor:", "").strip()

    # ── Component 1: Tumour Type Base Risk ────────────────────────────────────
    base_risk = TUMOR_TYPE_BASE_RISK.get(tumor_key, 30)
    # Modulate by confidence
    c1 = base_risk * (confidence / 100)

    # ── Component 2: Tumour Size ──────────────────────────────────────────────
    # Area > 20% → high penalty
    size_factor = np.clip(area_pct / 25.0, 0, 1) * 25
    c2 = float(size_factor)

    # ── Component 3: Clinical Factors ─────────────────────────────────────────
    c3 = 0.0
    c3 += 10 if prior_tumor   else 0
    c3 +=  5 if family_history else 0
    # Age factor
    if patient_age >= 60:
        c3 += 10
    elif patient_age >= 45:
        c3 += 5
    elif patient_age < 15:
        c3 += 3   # paediatric — elevated concern
    # Gender (slight statistical modulation)
    if patient_gender.lower() == "male" and tumor_key == "glioma":
        c3 += 3
    elif patient_gender.lower() == "female" and tumor_key == "meningioma":
        c3 += 2
    # Symptoms
    for sym in symptoms:
        c3 += SYMPTOM_RISK_MAP.get(sym, 0)
    c3 = float(np.clip(c3, 0, 30))

    # ── Component 4: Radiomics ────────────────────────────────────────────────
    c4 = 0.0
    if radiomics_features:
        entropy    = radiomics_features.get("firstorder_Entropy", 0) or 0
        homogeneity= radiomics_features.get("glcm_Homogeneity",  1) or 1
        contrast_r = radiomics_features.get("glcm_Contrast",     0) or 0
        # High entropy + low homogeneity = irregular texture = higher risk
        c4 = float(np.clip((entropy * 5) + (1 - homogeneity) * 10 + contrast_r * 2, 0, 20))

    # ── Component 5: MRI Quality Penalty ─────────────────────────────────────
    # Poor quality reduces confidence in all other scores → small risk boost
    quality_penalty = float(np.clip((100 - mri_quality) / 100 * 5, 0, 5))
    c5 = quality_penalty

    # ── Aggregate ─────────────────────────────────────────────────────────────
    raw_score   = c1 + c2 + c3 + c4 + c5
    risk_score  = float(np.clip(raw_score, 0, 100))

    if risk_score < 35:
        risk_level = "Low"
        risk_color = "#10b981"
    elif risk_score < 65:
        risk_level = "Medium"
        risk_color = "#f59e0b"
    else:
        risk_level = "High"
        risk_color = "#ef4444"

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_risk_summary(tumor_key, risk_level, risk_score, c1, c2, c3, c4)

    return {
        "risk_score":     round(risk_score, 1),
        "risk_level":     risk_level,
        "risk_color":     risk_color,
        "components": {
            "tumor_type_risk":    round(c1, 1),
            "size_risk":          round(c2, 1),
            "clinical_risk":      round(c3, 1),
            "radiomics_risk":     round(c4, 1),
            "quality_penalty":    round(c5, 1),
        },
        "summary": summary,
    }


# ─── Phase I: NeuroScore System ───────────────────────────────────────────────

def compute_neuroscore(
    tumor_type:           str,
    confidence:           float,
    area_pct:             float,
    mri_quality_score:    float,
    radiomics_features:   Dict[str, Any],
    uncertainty_pct:      float = 5.0,
    patient_age:          int   = 0,
    symptoms:             List[str] = None,
    family_history:       bool  = False,
    prior_tumor:          bool  = False,
) -> Dict[str, Any]:
    """
    Proprietary NeuroScore — the primary novelty metric.

    Combines 6 weighted dimensions into a single 0-100 score.

    Scoring rubric (weights):
    ┌──────────────────────────────┬────────┐
    │ Dimension                    │ Weight │
    ├──────────────────────────────┼────────┤
    │ Tumor Type Severity          │  0.25  │
    │ Model Confidence             │  0.20  │
    │ Tumor Size (area %)          │  0.15  │
    │ MRI Quality                  │  0.10  │
    │ Radiomics Complexity         │  0.15  │
    │ Clinical Risk Factors        │  0.15  │
    └──────────────────────────────┴────────┘
    """
    if symptoms is None:
        symptoms = []

    tumor_key = tumor_type.lower().replace(" ", "").replace("tumor:", "").strip()

    # ── D1: Tumour Type Severity (0-100) ─────────────────────────────────────
    severity_map = {"glioma": 90, "meningioma": 55, "pituitary": 40, "notumor": 5}
    d1 = float(severity_map.get(tumor_key, 30))

    # ── D2: Model Confidence weighted by uncertainty ──────────────────────────
    # High confidence + low uncertainty → high score (if tumour)
    # High confidence + no tumour → low score
    conf_factor  = confidence / 100
    unc_penalty  = min(uncertainty_pct / 50, 0.3)     # max 30% penalty
    if tumor_key == "notumor":
        d2 = 5.0                                        # healthy = very low score
    else:
        d2 = (conf_factor - unc_penalty) * 100
    d2 = float(np.clip(d2, 0, 100))

    # ── D3: Tumour Size (0-100) ───────────────────────────────────────────────
    d3 = float(np.clip(area_pct * 4, 0, 100))          # 25% area → max score

    # ── D4: MRI Quality — inversely weighted ─────────────────────────────────
    # Good quality = reliable score; poor quality = score penalised
    d4_quality = mri_quality_score                      # 0-100
    quality_reliability = d4_quality / 100              # 0-1

    # ── D5: Radiomics Complexity (0-100) ─────────────────────────────────────
    d5 = 0.0
    if radiomics_features:
        entropy     = float(radiomics_features.get("firstorder_Entropy",    5) or 5)
        homogeneity = float(radiomics_features.get("glcm_Homogeneity",      0.5) or 0.5)
        contrast_r  = float(radiomics_features.get("glcm_Contrast",         0) or 0)
        # Normalise: entropy [0-8], homogeneity [0-1], contrast [0-20]
        e_norm = np.clip(entropy  / 8.0, 0, 1)
        h_norm = 1 - homogeneity                        # lower homogeneity = more complex
        c_norm = np.clip(contrast_r / 20.0, 0, 1)
        d5 = float((e_norm * 0.4 + h_norm * 0.4 + c_norm * 0.2) * 100)

    # ── D6: Clinical Risk Factors (0-100) ─────────────────────────────────────
    d6 = 0.0
    d6 += 15 if prior_tumor    else 0
    d6 += 10 if family_history else 0
    if patient_age >= 60:  d6 += 15
    elif patient_age >= 45: d6 += 8
    elif patient_age < 15:  d6 += 5
    for sym in symptoms:
        d6 += SYMPTOM_RISK_MAP.get(sym, 0) * 1.5
    d6 = float(np.clip(d6, 0, 100))

    # ── Weighted Sum ──────────────────────────────────────────────────────────
    WEIGHTS = {
        "tumor_type":    0.25,
        "confidence":    0.20,
        "tumor_size":    0.15,
        "mri_quality":   0.10,
        "radiomics":     0.15,
        "clinical":      0.15,
    }

    raw = (
        WEIGHTS["tumor_type"]  * d1
        + WEIGHTS["confidence"]  * d2
        + WEIGHTS["tumor_size"]  * d3
        + WEIGHTS["mri_quality"] * d4_quality
        + WEIGHTS["radiomics"]   * d5
        + WEIGHTS["clinical"]    * d6
    )

    # Apply quality reliability scaling
    # If quality is very low, overall NeuroScore is less reliable
    scaled = raw * (0.7 + 0.3 * quality_reliability)
    neuroscore = float(np.clip(scaled, 0, 100))

    if neuroscore < 25:
        ns_level, ns_color = "Low Risk",    "#10b981"
    elif neuroscore < 55:
        ns_level, ns_color = "Medium Risk", "#f59e0b"
    elif neuroscore < 75:
        ns_level, ns_color = "High Risk",   "#ef4444"
    else:
        ns_level, ns_color = "Critical",    "#dc2626"

    return {
        "neuroscore":  round(neuroscore, 1),
        "level":       ns_level,
        "color":       ns_color,
        "dimensions": {
            "tumor_severity":       round(d1, 1),
            "model_confidence":     round(d2, 1),
            "tumor_size":           round(d3, 1),
            "mri_quality":          round(d4_quality, 1),
            "radiomics_complexity": round(d5, 1),
            "clinical_factors":     round(d6, 1),
        },
        "weights": WEIGHTS,
        "interpretation": _interpret_neuroscore(neuroscore, ns_level, tumor_key),
    }


def _interpret_neuroscore(score: float, level: str, tumor_key: str) -> str:
    base = f"NeuroScore {score:.1f}/100 — {level}. "
    if tumor_key == "notumor":
        return base + ("No evidence of neoplasm. Routine follow-up is recommended "
                       "if clinical symptoms persist.")
    if score < 25:
        return base + ("Minimal concern. The detected abnormality presents low-risk "
                       "radiological and clinical features. Annual surveillance advised.")
    if score < 55:
        return base + ("Moderate concern. Multiparametric follow-up within 3–6 months "
                       "recommended. Neurosurgical consultation may be warranted.")
    if score < 75:
        return base + ("High concern. Immediate specialist referral recommended. "
                       "Advanced MRI sequences (perfusion, spectroscopy) advised.")
    return base + ("Critical concern. Urgent neurosurgical evaluation required. "
                   "This NeuroScore indicates features associated with high-grade "
                   "or aggressively growing neoplasm.")


def _build_risk_summary(tumor_key, level, score, c1, c2, c3, c4) -> str:
    if tumor_key == "notumor":
        return (f"Risk Score {score:.0f}/100 ({level}). No tumour detected. "
                "Standard clinical surveillance protocol advised.")
    drivers = []
    if c1 > 40: drivers.append("tumour type severity")
    if c2 > 12: drivers.append("significant tumour size")
    if c3 > 10: drivers.append("clinical risk factors")
    if c4 > 8:  drivers.append("complex radiomics texture")
    driver_str = ", ".join(drivers) if drivers else "multiple low-level factors"
    return (f"Risk Score {score:.0f}/100 ({level}). "
            f"Primary risk drivers: {driver_str}. "
            f"Multidisciplinary tumour board review recommended.")
