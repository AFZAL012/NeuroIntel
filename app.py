"""
NeuroIntel 3.0 — Main Flask Application
Complete CDSS with 14 research-grade modules.
Preserves all existing functionality from main.py.
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import base64
import traceback
from datetime import datetime
from typing import Any

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, send_file, session,
)
from tensorflow.keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_PATH, IMAGE_SIZE, CLASS_LABELS, CLASS_MAP,
    UPLOAD_FOLDER, EXPORTS_FOLDER, SESSIONS_FOLDER,
    BENCHMARK_METRICS, MC_DROPOUT_ITERATIONS,
)

# ── Dynamic model metadata (written by train.py after each training run) ──────
_META_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

def _load_model_meta() -> dict:
    """Read model_metadata.json if present; fall back to legacy MobileNetV2 defaults."""
    if os.path.exists(_META_PATH):
        try:
            with open(_META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"model_name": "MobileNetV2", "input_size": IMAGE_SIZE}

def _resolve_preprocess_fn(model_name: str):
    """Return the correct Keras preprocessing function for *model_name*."""
    mn = (model_name or "").lower()
    if "efficientnet" in mn:
        from tensorflow.keras.applications.efficientnet import preprocess_input as pp
    elif "resnet" in mn:
        from tensorflow.keras.applications.resnet50 import preprocess_input as pp
    else:  # default / MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp
    return pp

_model_meta     = _load_model_meta()
_MODEL_NAME     = _model_meta.get("model_name", "MobileNetV2")
_INPUT_SIZE     = int(_model_meta.get("input_size",  IMAGE_SIZE))
preprocess_input = _resolve_preprocess_fn(_MODEL_NAME)

# ── Backend modules ───────────────────────────────────────────────────────────
from backend.quality.mri_quality         import assess_quality
from backend.segmentation.segmentation   import segment_tumor_opencv
from backend.localization.localization   import localize_tumor, draw_bounding_box
from backend.radiomics.radiomics_extractor import extract_radiomics
from backend.uncertainty.mc_dropout      import estimate_uncertainty, add_mc_dropout_to_model
from backend.explainability.grad_cam     import run_xai_pipeline, _find_last_conv
from backend.neuroscore.neuroscore       import compute_risk, compute_neuroscore
from backend.longitudinal.longitudinal   import (
    get_or_create_session, add_scan, compute_longitudinal, clear_session,
)
from backend.research.hybrid_model       import get_architecture_description, get_benchmark_comparison
from backend.research.paper_generator    import generate_paper_draft
from backend.reports.pdf_generator       import generate_pdf_report
from backend.validation.mri_validator    import validate_mri

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("NeuroIntel")

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

for folder in [UPLOAD_FOLDER, EXPORTS_FOLDER, SESSIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Model loading ─────────────────────────────────────────────────────────────
logger.info(f"Loading model from {MODEL_PATH} …")
_model = load_model(MODEL_PATH)
_mc_model = add_mc_dropout_to_model(_model)
logger.info("Model loaded successfully.")


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING ROUTES (preserved from main.py)
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    """Original prediction route — preserved for backward compatibility."""
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result, confidence = _predict_tumor(filepath)
            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence * 100:.2f}",
                file_path=f"/uploads/{file.filename}",
            )
    return render_template("index.html", result=None)


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/samples/<filename>")
def get_sample_file(filename):
    return send_from_directory(".", filename)


@app.route("/exports/<filename>")
def get_export_file(filename):
    return send_from_directory(EXPORTS_FOLDER, filename)


# ═════════════════════════════════════════════════════════════════════════════
# API ROUTES — JSON endpoints for AJAX frontend
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/full-analysis", methods=["POST"])
def api_full_analysis():
    """
    Master endpoint — runs the complete NeuroIntel pipeline in one call.
    Returns all analysis results as a single JSON response.
    """
    t0 = time.perf_counter()

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    # Patient metadata (from form fields)
    patient_age      = int(request.form.get("age", 0) or 0)
    patient_gender   = request.form.get("gender", "unknown")
    symptoms         = request.form.getlist("symptoms")
    family_history   = request.form.get("family_history", "false").lower() == "true"
    prior_tumor      = request.form.get("prior_tumor",    "false").lower() == "true"
    patient_id       = request.form.get("patient_id", f"PT-{datetime.now().strftime('%H%M%S')}")
    scan_date        = request.form.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
    enable_xai       = request.form.get("enable_xai", "true").lower() == "true"
    session_id       = request.form.get("session_id") or session.get("session_id")

    # Save file
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    logger.info(f"Full analysis started: {filename}")

    try:
        # ── Stage 1: MRI Validation Layer ─────────────────────────────────────
        validation_res = validate_mri(filepath)
        if not validation_res["is_mri"]:
            return jsonify({
                "error": "invalid_mri",
                "message": "Invalid Input. Please upload a valid brain MRI scan.",
                "confidence": validation_res["confidence"]
            }), 422

        if validation_res["confidence"] < 0.85:
            return jsonify({
                "error": "invalid_mri_unverified",
                "message": "Invalid Input. Please upload a valid brain MRI scan.",
                "confidence": validation_res["confidence"]
            }), 422

        # ── Phase A: MRI Quality ──────────────────────────────────────────────
        quality = assess_quality(filepath)
        if quality["status"] == "rejected":
            return jsonify({
                "error": "poor_quality",
                "quality": quality,
                "message": quality["warnings"][0] if quality["warnings"] else "Image quality too poor.",
            }), 422

        # ── Core Prediction ───────────────────────────────────────────────────
        img_arr, img_pp = _prepare_image(filepath)

        # ── Phase F: Uncertainty (MC Dropout) ────────────────────────────────
        uncertainty = estimate_uncertainty(
            _mc_model, img_pp, CLASS_LABELS, n_iter=MC_DROPOUT_ITERATIONS
        )
        pred_idx   = uncertainty["predicted_index"]
        pred_label = uncertainty["predicted_class"]
        confidence = uncertainty["confidence_pct"]

        # Human-readable result (matches original main.py format)
        if pred_label == "notumor":
            result_str = "No Tumor"
        else:
            result_str = f"Tumor: {pred_label}"

        # ── Phase B: Segmentation ─────────────────────────────────────────────
        segmentation = segment_tumor_opencv(filepath)

        # ── Phase C: Localization ─────────────────────────────────────────────
        localization  = localize_tumor(filepath, segmentation.get("mask_b64"))
        loc_image_b64 = draw_bounding_box(filepath, localization)
        localization["bbox_image_b64"] = loc_image_b64

        # ── Phase D: Radiomics ────────────────────────────────────────────────
        radiomics = extract_radiomics(filepath, segmentation.get("mask_b64"))

        # ── Phase G: XAI ──────────────────────────────────────────────────────
        xai = {}
        if enable_xai:
            try:
                xai = run_xai_pipeline(_model, img_pp, pred_idx, image_path=filepath)
            except Exception as e:
                logger.warning(f"XAI failed: {e}")
                xai = {"error": str(e)}

        # ── Phase H: Risk Assessment ──────────────────────────────────────────
        risk = compute_risk(
            tumor_type=pred_label,
            confidence=confidence,
            area_pct=segmentation.get("area_pct", 0),
            volume_mm3=segmentation.get("volume_mm3", 0),
            mri_quality=quality["score"],
            radiomics_features=radiomics.get("features", {}),
            patient_age=patient_age,
            patient_gender=patient_gender,
            symptoms=symptoms,
            family_history=family_history,
            prior_tumor=prior_tumor,
        )

        # ── Phase I: NeuroScore ───────────────────────────────────────────────
        neuroscore = compute_neuroscore(
            tumor_type=pred_label,
            confidence=confidence,
            area_pct=segmentation.get("area_pct", 0),
            mri_quality_score=quality["score"],
            radiomics_features=radiomics.get("features", {}),
            uncertainty_pct=uncertainty.get("uncertainty_pct", 5),
            patient_age=patient_age,
            symptoms=symptoms,
            family_history=family_history,
            prior_tumor=prior_tumor,
        )

        # ── Phase J: Longitudinal ─────────────────────────────────────────────
        session_id = get_or_create_session(session_id)
        session["session_id"] = session_id
        longitudinal = add_scan(
            session_id=session_id,
            scan_date=scan_date,
            filename=filename,
            tumor_type=pred_label,
            area_pct=segmentation.get("area_pct", 0),
            volume_mm3=segmentation.get("volume_mm3", 0),
            neuroscore=neuroscore["neuroscore"],
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"Analysis complete in {elapsed_ms} ms — {result_str} ({confidence:.1f}%)")

        return jsonify({
            "success":     True,
            "file_path":   f"/uploads/{filename}",
            "result":      result_str,
            "confidence":  round(confidence, 2),
            "class_index": pred_idx,
            "class_label": pred_label,
            "processing_ms": elapsed_ms,
            # All modules
            "quality":      quality,
            "uncertainty":  uncertainty,
            "segmentation": segmentation,
            "localization": localization,
            "radiomics":    radiomics,
            "xai":          xai,
            "risk":         risk,
            "neuroscore":   neuroscore,
            "longitudinal": longitudinal,
            "patient_info": {
                "patient_id":     patient_id,
                "age":            patient_age,
                "gender":         patient_gender,
                "symptoms":       symptoms,
                "family_history": family_history,
                "prior_tumor":    prior_tumor,
                "scan_date":      scan_date,
            },
        })

    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Individual module endpoints (for progressive enhancement) ─────────────────

@app.route("/api/quality", methods=["POST"])
def api_quality():
    filepath = _get_uploaded_path(request)
    if isinstance(filepath, tuple):
        return filepath
    return jsonify(assess_quality(filepath))


@app.route("/api/segmentation", methods=["POST"])
def api_segmentation():
    filepath = _get_uploaded_path(request)
    if isinstance(filepath, tuple):
        return filepath
    return jsonify(segment_tumor_opencv(filepath))


@app.route("/api/xai", methods=["POST"])
def api_xai():
    filepath = _get_uploaded_path(request)
    if isinstance(filepath, tuple):
        return filepath
    class_index = int(request.form.get("class_index", 1))
    _, img_pp = _prepare_image(filepath)
    result = run_xai_pipeline(_model, img_pp, class_index)
    return jsonify(result)


@app.route("/api/report", methods=["POST"])
def api_report():
    """Generate and return a clinical PDF report."""
    try:
        data = request.get_json(force=True)
        patient_info  = data.get("patient_info", {})
        prediction    = data.get("prediction",   {})
        quality       = data.get("quality",      {})
        segmentation  = data.get("segmentation", {})
        localization  = data.get("localization", {})
        radiomics     = data.get("radiomics",    {})
        neuroscore    = data.get("neuroscore",   {})
        risk          = data.get("risk",         {})
        xai           = data.get("xai",          {})
        uncertainty   = data.get("uncertainty",  {})
        image_path    = os.path.join(UPLOAD_FOLDER, data.get("filename", ""))

        pdf_bytes = generate_pdf_report(
            patient_info, prediction, quality, segmentation,
            localization, radiomics, neuroscore, risk,
            xai, uncertainty, image_path,
        )

        import io
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return send_file(
            buf, mimetype="application/pdf",
            as_attachment=True,
            download_name=f"neurointel_report_{patient_info.get('patient_id', 'patient')}.pdf",
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/paper", methods=["GET"])
def api_paper():
    """Generate and return research paper draft as Markdown."""
    arch    = get_architecture_description()
    metrics = BENCHMARK_METRICS
    paper   = generate_paper_draft(metrics, {}, arch)
    import io
    buf = io.BytesIO(paper.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf, mimetype="text/markdown",
        as_attachment=True,
        download_name="neurointel_research_paper_draft.md",
    )


@app.route("/api/longitudinal", methods=["GET"])
def api_longitudinal():
    sid = request.args.get("session_id") or session.get("session_id")
    if not sid:
        return jsonify({"scans": [], "trend": "no_session"})
    return jsonify(compute_longitudinal(sid))


@app.route("/api/longitudinal/clear", methods=["POST"])
def api_longitudinal_clear():
    sid = request.json.get("session_id") or session.get("session_id")
    if sid:
        clear_session(sid)
    return jsonify({"success": True})


@app.route("/api/research/architecture", methods=["GET"])
def api_architecture():
    return jsonify(get_architecture_description())


@app.route("/api/research/benchmark", methods=["GET"])
def api_benchmark():
    return jsonify(BENCHMARK_METRICS)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status":     "online",
        "model":      _MODEL_NAME,
        "input_size": _INPUT_SIZE,
        "version":    "NeuroIntel 3.0",
        "timestamp":  datetime.now().isoformat(),
    })


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _predict_tumor(image_path: str):
    """Original prediction function — uses dynamic input size from metadata."""
    img      = load_img(image_path, target_size=(_INPUT_SIZE, _INPUT_SIZE))
    arr      = img_to_array(img)
    arr      = preprocess_input(arr)
    arr      = np.expand_dims(arr, axis=0)
    preds    = _model.predict(arr, verbose=0)
    idx      = int(np.argmax(preds))
    conf     = float(np.max(preds))
    label    = CLASS_LABELS[idx]
    result   = "No Tumor" if label == "notumor" else f"Tumor: {label}"
    return result, conf


def _prepare_image(image_path: str):
    """Load + preprocess image for model inference — uses dynamic input size."""
    img     = load_img(image_path, target_size=(_INPUT_SIZE, _INPUT_SIZE))
    arr     = img_to_array(img)
    pp      = preprocess_input(arr.copy())
    return arr, np.expand_dims(pp, axis=0)


def _get_uploaded_path(req):
    """Save uploaded file and return path, or return error tuple."""
    f = req.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("Starting NeuroIntel 3.0 on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
