"""
Phase A — MRI Quality Assessment
Evaluates blur, noise, contrast, brightness, and resolution.
Returns a 0-100 quality score with clinical warnings.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any


def assess_quality(image_path: str) -> Dict[str, Any]:
    """
    Analyse MRI image quality along five axes and return a structured report.

    Returns
    -------
    dict with keys:
        score          : float  0-100
        blur_score     : float  0-100 (100 = perfectly sharp)
        contrast_score : float  0-100
        noise_score    : float  0-100 (100 = low noise)
        brightness_score: float 0-100
        resolution_ok  : bool
        width, height  : int
        warnings       : list[str]
        suggestions    : list[str]
        status         : "accepted" | "warning" | "rejected"
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _error_result("Cannot read image file.")

    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    warnings:    list[str] = []
    suggestions: list[str] = []

    # ── 1. BLUR (Laplacian variance) ─────────────────────────────────────────
    lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Empirically: < 50 = very blurry, > 500 = sharp
    blur_score = float(np.clip((lap_var / 500.0) * 100, 0, 100))
    if blur_score < 30:
        warnings.append("Image is severely blurred.")
        suggestions.append("Use a higher-resolution MRI scan or reduce motion artefacts.")
    elif blur_score < 55:
        warnings.append("Mild blurring detected.")
        suggestions.append("Consider using a scan with better field-of-view sharpness.")

    # ── 2. CONTRAST (RMS contrast) ────────────────────────────────────────────
    mean_val      = float(np.mean(gray))
    rms_contrast  = float(np.sqrt(np.mean((gray.astype(np.float64) - mean_val) ** 2)))
    # Normalise to 0-100 (max meaningful RMS ~ 80)
    contrast_score = float(np.clip((rms_contrast / 80.0) * 100, 0, 100))
    if contrast_score < 20:
        warnings.append("Very low image contrast detected.")
        suggestions.append("Adjust window/level settings or use contrast-enhanced MRI.")
    elif contrast_score < 40:
        warnings.append("Sub-optimal contrast. Tumour boundaries may be unclear.")
        suggestions.append("Enhance contrast pre-processing or verify scan protocol.")

    # ── 3. NOISE (MAD on smooth regions) ─────────────────────────────────────
    blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
    residual   = gray.astype(np.float64) - blurred.astype(np.float64)
    mad_noise  = float(np.mean(np.abs(residual)))
    # Lower MAD → less noise; clip at 20 for scoring
    noise_score = float(np.clip((1.0 - mad_noise / 20.0) * 100, 0, 100))
    if noise_score < 30:
        warnings.append("High noise level detected.")
        suggestions.append("Apply denoising filters or use a higher SNR scanner.")
    elif noise_score < 55:
        warnings.append("Moderate noise present.")

    # ── 4. BRIGHTNESS ─────────────────────────────────────────────────────────
    # Ideal MRI brightness: 80–180 out of 255
    if mean_val < 40:
        brightness_score = mean_val / 40.0 * 50   # very dark
        warnings.append("Image is very dark — possible underexposure.")
        suggestions.append("Check scanner gain settings or windowing parameters.")
    elif mean_val > 220:
        brightness_score = max(0.0, (255 - mean_val) / 35.0 * 50)
        warnings.append("Image is overexposed — tissue detail may be lost.")
        suggestions.append("Reduce scan brightness or apply histogram equalisation.")
    else:
        # Parabola peaking at 130
        brightness_score = 100 - ((mean_val - 130) ** 2) / 200.0
        brightness_score = float(np.clip(brightness_score, 60, 100))

    # ── 5. RESOLUTION ─────────────────────────────────────────────────────────
    MIN_DIM      = 128
    resolution_ok = (w >= MIN_DIM and h >= MIN_DIM)
    if not resolution_ok:
        warnings.append(f"Image resolution {w}×{h} is below minimum {MIN_DIM}×{MIN_DIM}.")
        suggestions.append("Use a minimum 128×128 px MRI image for reliable analysis.")

    # ── Aggregate Score ───────────────────────────────────────────────────────
    res_factor = 100.0 if resolution_ok else 30.0
    score = (
        0.30 * blur_score
        + 0.25 * contrast_score
        + 0.25 * noise_score
        + 0.10 * brightness_score
        + 0.10 * res_factor
    )
    score = float(np.clip(score, 0, 100))

    if score < 20:
        status = "rejected"
        warnings.insert(0, "⛔ Scan quality is too poor for reliable AI analysis.")
    elif score < 45:
        status = "warning"
    else:
        status = "accepted"

    return {
        "score":            round(score, 1),
        "blur_score":       round(blur_score, 1),
        "contrast_score":   round(contrast_score, 1),
        "noise_score":      round(noise_score, 1),
        "brightness_score": round(brightness_score, 1),
        "resolution_ok":    resolution_ok,
        "width":            w,
        "height":           h,
        "warnings":         warnings,
        "suggestions":      suggestions,
        "status":           status,
    }


def _error_result(msg: str) -> Dict[str, Any]:
    return {
        "score": 0.0, "blur_score": 0.0, "contrast_score": 0.0,
        "noise_score": 0.0, "brightness_score": 0.0,
        "resolution_ok": False, "width": 0, "height": 0,
        "warnings": [msg], "suggestions": [], "status": "rejected",
    }
