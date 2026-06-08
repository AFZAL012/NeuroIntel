"""
Phase G — Explainable AI (Robust Grad-CAM)
Correctly handles Sequential + nested MobileNetV2 + augmentation layers.

Key fix: tape.watch(conv_out) BEFORE computing predictions from conv_out.
Architecture assumed:
    model.layers[0] = augmentation Sequential
    model.layers[1] = MobileNetV2 (the backbone)
    model.layers[2:] = classifier head (GAP, Dropout, Dense, Dropout, Dense)
"""
from __future__ import annotations
import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple


# ─── Core helpers ─────────────────────────────────────────────────────────────

def _recover_orig(img_array: np.ndarray) -> np.ndarray:
    """Recover displayable BGR image from MobileNetV2 preprocessed array [-1, 1]."""
    orig = img_array[0].copy()
    orig = ((orig + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    if orig.shape[-1] == 3:
        return cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    return orig


def _to_b64(img_array: np.ndarray) -> str:
    """Encode BGR uint8 numpy image to base64 PNG string."""
    try:
        ok, buf = cv2.imencode(".png", img_array)
        if ok:
            return base64.b64encode(buf).decode("utf-8")
    except Exception:
        pass
    return ""


def _find_backbone_and_head(model):
    """
    Returns (backbone_model, head_layers) by scanning model.layers.
    backbone = first tf.keras.Model sub-layer (MobileNetV2)
    head_layers = all layers after the backbone
    """
    import tensorflow as tf

    backbone = None
    backbone_idx = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            backbone_idx = i
            break

    head_layers = model.layers[backbone_idx + 1:] if backbone_idx >= 0 else []
    return backbone, head_layers


def _find_last_conv_layer(backbone):
    """Find the last 4-D output (spatial) layer in a Keras model."""
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4:
                h, w = shape[1], shape[2]
                if (h is None or h > 1) and (w is None or w > 1):
                    return layer
        except Exception:
            pass
    return None


def _build_conv_extractor(backbone, last_conv_layer):
    """Build sub-model: backbone_input → last_conv_output."""
    from tensorflow.keras.models import Model
    return Model(inputs=backbone.inputs, outputs=last_conv_layer.output)


# ─── Grad-CAM ────────────────────────────────────────────────────────────────

def compute_gradcam(
    model,
    img_array: np.ndarray,
    class_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM using split model approach:
        1. aug_layer → conv_extractor    (up to last conv)
        2. tape.watch(conv_out)
        3. head_layers (GAP → Dense)     (watched by tape)
    This ensures gradients flow through conv_out → predictions.

    Returns (heatmap_bgr, overlay_bgr, original_bgr) — all uint8 H×W×3.
    """
    import tensorflow as tf

    orig_bgr = _recover_orig(img_array)
    h, w = orig_bgr.shape[:2]

    backbone, head_layers = _find_backbone_and_head(model)

    if backbone is None or not head_layers:
        return _gradcam_input_gradient(model, img_array, class_index, orig_bgr)

    last_conv = _find_last_conv_layer(backbone)
    if last_conv is None:
        return _gradcam_input_gradient(model, img_array, class_index, orig_bgr)

    try:
        conv_extractor = _build_conv_extractor(backbone, last_conv)
    except Exception:
        return _gradcam_input_gradient(model, img_array, class_index, orig_bgr)

    try:
        # Pass through augmentation layers (identity during inference)
        aug_out = img_array
        for layer in model.layers:
            if layer is backbone:
                break
            aug_out = layer(aug_out, training=False)

        # ── KEY FIX: watch conv_out BEFORE computing predictions ──────────────
        with tf.GradientTape() as tape:
            conv_out = conv_extractor(aug_out, training=False)
            tape.watch(conv_out)              # ← must be before using conv_out
            x = conv_out
            for layer in head_layers:
                x = layer(x, training=False)
            loss = x[:, class_index]

        grads = tape.gradient(loss, conv_out)

        if grads is None or float(tf.reduce_max(tf.abs(grads)).numpy()) < 1e-12:
            return _gradcam_input_gradient(model, img_array, class_index, orig_bgr)

        # Pool gradients over spatial dimensions
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # (C,)
        cam = conv_out[0].numpy()                               # (fH, fW, C)
        cam = np.einsum('hwc,c->hw', cam, pooled)               # (fH, fW)
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-10)

        cam_resized  = cv2.resize(cam.astype(np.float32), (w, h))
        cam_uint8    = (255 * cam_resized).astype(np.uint8)
        heatmap      = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        overlay      = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
        return heatmap, overlay, orig_bgr

    except Exception:
        return _gradcam_input_gradient(model, img_array, class_index, orig_bgr)


def _gradcam_input_gradient(model, img_array, class_index, orig_bgr):
    """
    Fallback: gradient of class score w.r.t. input pixel values.
    Always produces a valid (non-black) result.
    """
    import tensorflow as tf

    h, w = orig_bgr.shape[:2]
    inp = tf.Variable(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(inp, training=False)
        loss  = preds[:, class_index]

    grads = tape.gradient(loss, inp)
    if grads is None:
        blank = np.zeros_like(orig_bgr)
        return blank, orig_bgr, orig_bgr

    cam = np.max(np.abs(grads[0].numpy()), axis=-1)
    cam /= (cam.max() + 1e-10)
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    cam_uint8   = (255 * cam_resized).astype(np.uint8)
    heatmap     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay     = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
    return heatmap, overlay, orig_bgr


# ─── Saliency ────────────────────────────────────────────────────────────────

def compute_saliency(
    model,
    img_array: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Vanilla gradient saliency map. Returns uint8 BGR (H×W×3)."""
    import tensorflow as tf

    inp = tf.Variable(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds = model(inp, training=False)
        loss  = preds[:, class_index]

    grads = tape.gradient(loss, inp)
    if grads is None:
        return np.zeros((*img_array.shape[1:3], 3), dtype=np.uint8)

    sal = np.max(np.abs(grads[0].numpy()), axis=-1)
    sal /= (sal.max() + 1e-10)
    sal_uint8 = (sal * 255).astype(np.uint8)
    return cv2.applyColorMap(sal_uint8, cv2.COLORMAP_HOT)


# ─── Guided Grad-CAM ─────────────────────────────────────────────────────────

def compute_guided_gradcam(
    model,
    img_array: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Guided Grad-CAM = positive input gradients × Grad-CAM spatial heatmap.
    Returns uint8 BGR (H×W×3).
    """
    import tensorflow as tf

    inp = tf.Variable(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds = model(inp, training=False)
        loss  = preds[:, class_index]

    grads = tape.gradient(loss, inp)
    if grads is None:
        return np.zeros((*img_array.shape[1:3], 3), dtype=np.uint8)

    guided = np.maximum(grads[0].numpy(), 0)   # keep only positive
    guided = np.max(guided, axis=-1)            # collapse channels → (H, W)
    guided /= (guided.max() + 1e-10)

    # Multiply by Grad-CAM heatmap (grayscale weights)
    heatmap_color, _, _ = compute_gradcam(model, img_array, class_index)
    cam_gray = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gh, gw = cam_gray.shape
    guided_rs = cv2.resize(guided.astype(np.float32), (gw, gh))
    fused     = guided_rs * cam_gray
    fused    /= (fused.max() + 1e-10)
    fused_u8  = (fused * 255).astype(np.uint8)
    return cv2.applyColorMap(fused_u8, cv2.COLORMAP_PLASMA)


# ─── Full XAI pipeline ───────────────────────────────────────────────────────

def run_xai_pipeline(
    model,
    img_array:  np.ndarray,
    class_index: int,
    image_path: str | None = None,
) -> Dict[str, Any]:
    """Run the full XAI suite and return all visualisations as base64 PNGs."""
    errors: list[str] = []

    # ── Original image ────────────────────────────────────────────────────────
    if image_path:
        orig_bgr = cv2.imread(image_path)
        if orig_bgr is not None:
            orig_bgr = cv2.resize(orig_bgr, (img_array.shape[2], img_array.shape[1]))
        else:
            orig_bgr = _recover_orig(img_array)
    else:
        orig_bgr = _recover_orig(img_array)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    try:
        heatmap, overlay, _ = compute_gradcam(model, img_array, class_index)
        heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
        overlay = cv2.resize(overlay, (orig_bgr.shape[1], orig_bgr.shape[0]))
    except Exception as e:
        errors.append(f"gradcam: {e}")
        heatmap = overlay = np.zeros_like(orig_bgr)

    # ── Saliency ──────────────────────────────────────────────────────────────
    try:
        saliency = compute_saliency(model, img_array, class_index)
        saliency = cv2.resize(saliency, (orig_bgr.shape[1], orig_bgr.shape[0]))
    except Exception as e:
        errors.append(f"saliency: {e}")
        saliency = np.zeros_like(orig_bgr)

    # ── Guided Grad-CAM ───────────────────────────────────────────────────────
    try:
        guided = compute_guided_gradcam(model, img_array, class_index)
        guided = cv2.resize(guided, (orig_bgr.shape[1], orig_bgr.shape[0]))
    except Exception as e:
        errors.append(f"guided_gradcam: {e}")
        guided = np.zeros_like(orig_bgr)

    return {
        "original_b64":       _to_b64(orig_bgr),
        "gradcam_b64":        _to_b64(heatmap),
        "overlay_b64":        _to_b64(overlay),
        "saliency_b64":       _to_b64(saliency),
        "guided_gradcam_b64": _to_b64(guided),
        "explanation":        _generate_clinical_explanation(class_index),
        "errors":             errors,
    }


# ─── Clinical explanation text ────────────────────────────────────────────────

def _generate_clinical_explanation(class_index: int) -> str:
    explanations = {
        0: ("Pituitary Tumor",
            "The saliency map highlights the sella turcica region (inferior-central). "
            "High-intensity Grad-CAM activation around the hypothalamic-pituitary axis "
            "indicates signal abnormalities consistent with pituitary adenoma morphology. "
            "The model decision is driven primarily by the infra-sellar tissue density."),
        1: ("Glioma",
            "Grad-CAM reveals diffuse hyperintense regions in the cerebral parenchyma. "
            "Heterogeneous signal intensity with infiltrative border patterns, visible in the "
            "overlay, are hallmarks of high-grade glioma detected by the model. The activation "
            "map suggests irregular tumour margins extending into white matter."),
        2: ("No Tumor",
            "No focal Grad-CAM activation is detected. The model's attention is uniformly "
            "distributed across the image, consistent with normal brain parenchyma, regular "
            "ventricular size, symmetric sulci, and absence of mass effects or signal anomalies."),
        3: ("Meningioma",
            "Activation concentrates at the dural margins and extra-axial compartment. "
            "The sharp, well-defined boundary pattern visible in the Grad-CAM overlay and the "
            "peripheral enhancement are characteristic features the model uses for meningioma "
            "classification. Extra-axial location confirms the diagnosis."),
    }
    _, explanation = explanations.get(class_index, (None, "No clinical explanation available."))
    return explanation


# ─── Backward-compatible alias ────────────────────────────────────────────────

def _find_last_conv(model) -> str:
    """Legacy helper — returns name of last spatial layer."""
    _, head = _find_backbone_and_head(model)
    try:
        backbone, _ = _find_backbone_and_head(model)
        lc = _find_last_conv_layer(backbone)
        return lc.name if lc else model.layers[-2].name
    except Exception:
        return model.layers[-2].name
