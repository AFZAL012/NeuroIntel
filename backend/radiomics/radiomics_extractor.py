"""
Phase D — Radiomics Feature Extraction
Extracts texture, shape, intensity, and GLCM features from tumor ROI.
Falls back to OpenCV-based computation when PyRadiomics is unavailable.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any


def extract_radiomics(image_path: str, mask_b64: str | None = None) -> Dict[str, Any]:
    """
    Extract radiomics features from the tumour region.

    Tries PyRadiomics first; falls back to OpenCV/NumPy implementation.

    Returns
    -------
    dict with 'features' (dict), 'source' ("pyradiomics" | "opencv"), 'success'
    """
    # ── Attempt PyRadiomics ───────────────────────────────────────────────────
    try:
        result = _pyradiomics_extract(image_path, mask_b64)
        if result["success"]:
            return result
    except Exception:
        pass

    # ── Fallback: OpenCV / NumPy ──────────────────────────────────────────────
    return _opencv_extract(image_path, mask_b64)


# ──────────────────────────────────────────────────────────────────────────────

def _pyradiomics_extract(image_path: str, mask_b64: str | None) -> Dict[str, Any]:
    import radiomics
    from radiomics import featureextractor
    import SimpleITK as sitk
    import tempfile, os, base64

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Cannot read image for PyRadiomics.")
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Build mask
    mask_arr = _decode_or_build_mask(img_bgr, mask_b64)
    mask_bin = (mask_arr > 127).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        img_path  = os.path.join(tmp, "image.nrrd")
        msk_path  = os.path.join(tmp, "mask.nrrd")
        sitk.WriteImage(sitk.GetImageFromArray(gray.astype(np.int16)), img_path)
        sitk.WriteImage(sitk.GetImageFromArray(mask_bin.astype(np.int16)), msk_path)

        params = {"binWidth": 25, "resampledPixelSpacing": None,
                  "interpolator": "sitkBSpline", "verbose": False}
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=["Mean", "Variance", "Skewness", "Kurtosis",
                        "Energy", "Entropy", "RootMeanSquared"],
            glcm=["Contrast", "Correlation", "Homogeneity",
                  "ClusterShade", "Idn"],
            shape2D=["Sphericity", "Elongation", "Perimeter",
                     "PerimeterSurfaceRatio"],
        )
        result = extractor.execute(img_path, msk_path, label=1)

    features = {k.replace("original_", ""): float(v)
                for k, v in result.items()
                if not k.startswith("diagnostics_")}
    return {"features": features, "source": "pyradiomics", "success": True}


def _opencv_extract(image_path: str, mask_b64: str | None) -> Dict[str, Any]:
    """
    Full radiomics feature set implemented with NumPy and scikit-image equivalents.
    """
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis

    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        return {"features": {}, "source": "opencv", "success": False,
                "error": "Cannot read image."}

    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask_arr = _decode_or_build_mask(img_bgr, mask_b64)
    binary   = mask_arr > 127

    if binary.sum() < 10:
        binary = np.ones_like(gray, dtype=bool)

    roi = gray[binary].astype(np.float64)

    # ── First-Order Statistics ────────────────────────────────────────────────
    mean_v  = float(np.mean(roi))
    var_v   = float(np.var(roi))
    std_v   = float(np.std(roi))
    skew_v  = float(skew(roi)) if len(roi) > 2 else 0.0
    kurt_v  = float(kurtosis(roi)) if len(roi) > 3 else 0.0
    energy  = float(np.sum(roi ** 2))
    rms     = float(np.sqrt(np.mean(roi ** 2)))

    # Entropy from histogram
    hist, _ = np.histogram(roi.astype(np.uint8), bins=256, range=(0, 255))
    prob    = hist / (hist.sum() + 1e-10)
    entropy = float(-np.sum(prob * np.log2(prob + 1e-10)))

    # ── GLCM (Gray Level Co-occurrence Matrix) ────────────────────────────────
    roi_img      = (gray * binary.astype(np.uint8))
    roi_scaled   = (roi_img * 15 / 255).astype(np.uint8)  # reduce to 16 levels
    glcm         = graycomatrix(roi_scaled, distances=[1], angles=[0, np.pi/4, np.pi/2],
                                levels=16, symmetric=True, normed=True)
    contrast     = float(graycoprops(glcm, "contrast").mean())
    correlation  = float(graycoprops(glcm, "correlation").mean())
    homogeneity  = float(graycoprops(glcm, "homogeneity").mean())
    dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())
    asm          = float(graycoprops(glcm, "ASM").mean())

    # ── Shape Features ────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c         = max(contours, key=cv2.contourArea)
        area_px   = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        sphericity   = (4 * np.pi * area_px) / (perimeter ** 2 + 1e-10)
        solidity     = area_px / (hull_area + 1e-10)
        rect         = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        elongation   = min(rw, rh) / (max(rw, rh) + 1e-10)
        perim_surf   = perimeter / (area_px + 1e-10)
    else:
        area_px = perimeter = hull_area = 0.0
        sphericity = solidity = elongation = perim_surf = 0.0

    features = {
        # First-order
        "firstorder_Mean":           round(mean_v, 4),
        "firstorder_Variance":       round(var_v, 4),
        "firstorder_StdDev":         round(std_v, 4),
        "firstorder_Skewness":       round(skew_v, 4),
        "firstorder_Kurtosis":       round(kurt_v, 4),
        "firstorder_Energy":         round(energy, 2),
        "firstorder_Entropy":        round(entropy, 4),
        "firstorder_RootMeanSquared":round(rms, 4),
        # GLCM
        "glcm_Contrast":             round(contrast, 4),
        "glcm_Correlation":          round(correlation, 4),
        "glcm_Homogeneity":          round(homogeneity, 4),
        "glcm_Dissimilarity":        round(dissimilarity, 4),
        "glcm_ASM":                  round(asm, 6),
        # Shape
        "shape_AreaPx":              round(area_px, 1),
        "shape_Perimeter":           round(perimeter, 1),
        "shape_Sphericity":          round(sphericity, 4),
        "shape_Solidity":            round(solidity, 4),
        "shape_Elongation":          round(elongation, 4),
        "shape_PerimSurfRatio":      round(perim_surf, 6),
    }
    return {"features": features, "source": "opencv", "success": True}


def _decode_or_build_mask(img_bgr: np.ndarray, mask_b64: str | None) -> np.ndarray:
    if mask_b64:
        try:
            import base64
            data   = base64.b64decode(mask_b64)
            np_arr = np.frombuffer(data, np.uint8)
            mask   = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
        except Exception:
            pass
    # Build quick Otsu mask
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
