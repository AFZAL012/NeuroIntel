"""
Phase C — Tumor Localization
Detects tumor centroid, bounding box, occupancy ratio, and approximate brain region.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any, Optional


# 3×3 grid → brain region labels
REGION_MAP = {
    (0, 0): "Left Frontal Lobe",     (1, 0): "Frontal Lobe (Central)",
    (2, 0): "Right Frontal Lobe",    (0, 1): "Left Parietal Lobe",
    (1, 1): "Central / Corpus Callosum", (2, 1): "Right Parietal Lobe",
    (0, 2): "Left Temporal / Occipital", (1, 2): "Posterior Fossa / Cerebellum",
    (2, 2): "Right Temporal / Occipital",
}


def localize_tumor(
    image_path: str,
    mask_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Localize tumour from image (or pre-computed mask).

    Parameters
    ----------
    image_path : path to original MRI
    mask_b64   : optional base64 mask from segmentation module

    Returns
    -------
    dict with bbox, region_label, centroid, occupancy_ratio, confidence
    """
    import base64

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _loc_error("Cannot read image.")

    h, w = img_bgr.shape[:2]

    # ── Decode mask or re-generate ────────────────────────────────────────────
    if mask_b64:
        img_data = base64.b64decode(mask_b64)
        np_arr   = np.frombuffer(img_data, np.uint8)
        mask     = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = _quick_mask(img_bgr)
    else:
        mask = _quick_mask(img_bgr)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "bbox": None, "region_label": "Not Detected",
            "centroid_x": None, "centroid_y": None,
            "occupancy_ratio": 0.0, "confidence": 0.0,
            "success": False,
        }

    # Largest contour
    largest     = max(contours, key=cv2.contourArea)
    area        = cv2.contourArea(largest)
    x, y, bw, bh = cv2.boundingRect(largest)

    # ── Centroid via image moments ────────────────────────────────────────────
    M  = cv2.moments(largest)
    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + bw // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + bh // 2

    # ── Region classification (3×3 grid) ─────────────────────────────────────
    col = min(int(cx / w * 3), 2)
    row = min(int(cy / h * 3), 2)
    region_label = REGION_MAP.get((col, row), "Unknown Region")

    # ── Occupancy ratio ───────────────────────────────────────────────────────
    occupancy_ratio = round(area / (h * w) * 100, 2)

    # ── Confidence heuristic ──────────────────────────────────────────────────
    # Based on contour solidity and aspect ratio
    hull    = cv2.convexHull(largest)
    hull_a  = cv2.contourArea(hull)
    solidity = area / hull_a if hull_a > 0 else 0.0
    aspect   = bw / bh if bh > 0 else 1.0
    conf     = float(np.clip(solidity * 0.6 + (1 - abs(aspect - 1) * 0.2) * 0.4, 0, 1))

    return {
        "bbox": {"x": x, "y": y, "w": bw, "h": bh},
        "region_label":     region_label,
        "centroid_x":       cx,
        "centroid_y":       cy,
        "occupancy_ratio":  occupancy_ratio,
        "confidence":       round(conf * 100, 1),
        "success":          True,
    }


def draw_bounding_box(image_path: str, loc_result: Dict[str, Any]) -> str:
    """Draw bounding box and region label onto the image, return as base64 PNG."""
    import base64

    img  = cv2.imread(image_path)
    if img is None or not loc_result.get("success"):
        return ""

    bbox   = loc_result["bbox"]
    region = loc_result["region_label"]
    cx, cy = loc_result["centroid_x"], loc_result["centroid_y"]

    cv2.rectangle(img, (bbox["x"], bbox["y"]),
                  (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                  (0, 255, 255), 2)
    cv2.circle(img, (cx, cy), 5, (255, 100, 0), -1)
    cv2.putText(img, region, (bbox["x"], max(bbox["y"] - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def _quick_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh     = clahe.apply(gray)
    _, otsu = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def _loc_error(msg: str) -> Dict[str, Any]:
    return {
        "bbox": None, "region_label": "Error", "centroid_x": None,
        "centroid_y": None, "occupancy_ratio": 0.0, "confidence": 0.0,
        "success": False, "error": msg,
    }
