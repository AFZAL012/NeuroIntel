"""
Phase B — Dual-Stage Tumor Segmentation
Stage 1: OpenCV morphological segmentation (immediate, no GPU)
Stage 2: U-Net architecture stub (ready for BraTS weight loading)
"""
from __future__ import annotations
import cv2
import numpy as np
import base64
from typing import Dict, Any


# ── Stage 1: OpenCV Morphological Segmentation ────────────────────────────────

def segment_tumor_opencv(image_path: str) -> Dict[str, Any]:
    """
    Perform tumor segmentation using Otsu thresholding + morphological ops.

    Returns
    -------
    dict with keys:
        mask_b64    : str  — base64 PNG of binary mask
        overlay_b64 : str  — base64 PNG of coloured overlay on original
        contour_b64 : str  — base64 PNG of boundary-only overlay
        area_px     : int  — tumour area in pixels
        area_pct    : float — tumour area as % of total image
        volume_mm3  : float — estimated volume (1 mm isotropic)
        centroid_x  : int | None
        centroid_y  : int | None
        success     : bool
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _seg_error("Cannot read image.")

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape

    # ── Pre-processing ────────────────────────────────────────────────────────
    denoised  = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(denoised)

    # ── Thresholding ──────────────────────────────────────────────────────────
    _, otsu   = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Morphological clean-up ────────────────────────────────────────────────
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened    = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  kernel, iterations=2)
    closed    = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Extract largest connected component (tumour candidate) ────────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    mask = np.zeros_like(gray)
    centroid_x, centroid_y = None, None

    if n_labels > 1:
        # Sort by area, skip background (label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        mask[labels == largest] = 255
        centroid_x = int(centroids[largest][0])
        centroid_y = int(centroids[largest][1])

    # ── Build visualisations ──────────────────────────────────────────────────
    # Coloured overlay
    overlay = img_bgr.copy()
    overlay[mask > 0] = [0, 64, 255]          # warm red channel
    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

    # Contour-only outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img  = img_bgr.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)

    # ── Metrics ───────────────────────────────────────────────────────────────
    area_px  = int(np.sum(mask > 0))
    area_pct = round(area_px / (h * w) * 100, 2)
    # Approximate 3-D volume: assume 1 mm² pixel, 3 mm slice thickness
    volume_mm3 = round(area_px * 3.0, 1)

    return {
        "mask_b64":    _to_b64(mask),
        "overlay_b64": _to_b64(blended),
        "contour_b64": _to_b64(contour_img),
        "area_px":     area_px,
        "area_pct":    area_pct,
        "volume_mm3":  volume_mm3,
        "centroid_x":  centroid_x,
        "centroid_y":  centroid_y,
        "success":     True,
    }


# ── Stage 2: U-Net Architecture Definition ────────────────────────────────────
# Import TF lazily to avoid startup delay when only Stage 1 is used

def build_unet(input_shape: tuple = (256, 256, 1), num_classes: int = 1):
    """
    Return a compiled U-Net model (weights not loaded — ready for BraTS training).
    Architecture: 4-level encoder-decoder with skip connections.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        def conv_block(x, filters, name):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                              name=f"{name}_c1")(x)
            x = layers.BatchNormalization(name=f"{name}_bn1")(x)
            x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                              name=f"{name}_c2")(x)
            x = layers.BatchNormalization(name=f"{name}_bn2")(x)
            return x

        inputs = layers.Input(shape=input_shape, name="input")

        # Encoder
        c1 = conv_block(inputs, 64,  "enc1"); p1 = layers.MaxPooling2D()(c1)
        c2 = conv_block(p1,    128,  "enc2"); p2 = layers.MaxPooling2D()(c2)
        c3 = conv_block(p2,    256,  "enc3"); p3 = layers.MaxPooling2D()(c3)
        c4 = conv_block(p3,    512,  "enc4"); p4 = layers.MaxPooling2D()(c4)

        # Bottleneck
        bn = conv_block(p4, 1024, "bottleneck")

        # Decoder
        u6 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bn)
        u6 = layers.concatenate([u6, c4]); d6 = conv_block(u6, 512, "dec4")
        u7 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d6)
        u7 = layers.concatenate([u7, c3]); d7 = conv_block(u7, 256, "dec3")
        u8 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d7)
        u8 = layers.concatenate([u8, c2]); d8 = conv_block(u8, 128, "dec2")
        u9 = layers.Conv2DTranspose(64,  2, strides=2, padding="same")(d8)
        u9 = layers.concatenate([u9, c1]); d9 = conv_block(u9,  64, "dec1")

        outputs = layers.Conv2D(num_classes, 1,
                                activation="sigmoid" if num_classes == 1 else "softmax",
                                name="output")(d9)
        model = Model(inputs, outputs, name="UNet-BraTS")
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model
    except Exception as e:
        return None


def get_unet_summary() -> str:
    """Return architecture summary as a string (for Research Hub display)."""
    model = build_unet()
    if model is None:
        return "U-Net model could not be built (TensorFlow unavailable)."
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_b64(img_array: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_array)
    return base64.b64encode(buf).decode("utf-8")


def _seg_error(msg: str) -> Dict[str, Any]:
    return {
        "mask_b64": "", "overlay_b64": "", "contour_b64": "",
        "area_px": 0, "area_pct": 0.0, "volume_mm3": 0.0,
        "centroid_x": None, "centroid_y": None, "success": False,
        "error": msg,
    }
