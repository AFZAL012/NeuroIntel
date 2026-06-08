"""
NeuroIntel 3.0 — Research-Grade Training Pipeline 2.0
======================================================
Multi-model benchmarking | Two-phase fine-tuning | Full evaluation suite
Targets 85–95% test accuracy on 4-class brain-tumor MRI dataset.

Classes:  glioma | meningioma | notumor | pituitary
Training: 5,600 images (1,400 / class)  — balanced
Testing:  1,600 images (400  / class)   — balanced

Output artefacts
----------------
  results/dataset_stats.txt
  results/confusion_matrices/<model>.png
  results/roc_curves/<model>.png
  results/error_analysis/<model>_errors.png
  results/benchmark_table.txt
  models/model.h5
  models/model_metadata.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import warnings
import textwrap

# Ensure UTF-8 output on Windows terminals (cp1252 workaround)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # silence TF C++ info logs
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")                              # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

TRAIN_DIR      = "Training"
TEST_DIR       = "Testing"
MODELS_DIR     = "models"
RESULTS_DIR    = "results"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "model.h5")
META_SAVE_PATH  = os.path.join(MODELS_DIR, "model_metadata.json")

# Class index mapping — must match Flask app CLASS_LABELS order
CLASS_NAMES    = ["pituitary", "glioma", "notumor", "meningioma"]
FOLDER_MAP: Dict[str, int] = {
    "pituitary":  0,
    "glioma":     1,
    "notumor":    2,
    "meningioma": 3,
}

BATCH_SIZE     = 32
SEED           = 42
N_CLASSES      = 4
CV_FOLDS       = 5

# Per-model input sizes (larger = better features, slower on CPU)
MODEL_CONFIGS: Dict[str, Dict] = {
    "MobileNetV2":  {"input_size": 160, "unfreeze_top": 30},
    "EfficientNetB0": {"input_size": 224, "unfreeze_top": 30},
    "ResNet50":     {"input_size": 224, "unfreeze_top": 40},
}

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------------------
# DIRECTORY SETUP
# ------------------------------------------------------------------------------

for d in [
    MODELS_DIR,
    RESULTS_DIR,
    os.path.join(RESULTS_DIR, "confusion_matrices"),
    os.path.join(RESULTS_DIR, "roc_curves"),
    os.path.join(RESULTS_DIR, "error_analysis"),
]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# §1  DATASET ANALYSIS & VALIDATION
# ==============================================================================

def verify_images(folder: str) -> Tuple[int, List[str]]:
    """Check every file in *folder* with PIL.verify(); return (ok, bad_paths)."""
    ok_count, bad = 0, []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue
        try:
            with Image.open(fpath) as img:
                img.verify()
            ok_count += 1
        except (UnidentifiedImageError, Exception):
            bad.append(fpath)
    return ok_count, bad


def analyze_dataset() -> Dict:
    """Print and save dataset statistics. Returns counts dict."""
    lines = [
        "=" * 62,
        " NeuroIntel — Dataset Statistics Report",
        "=" * 62,
        f"{'Class':<14} {'Train':>8} {'Test':>8} {'Train%':>8} {'Test%':>8}",
        "-" * 62,
    ]

    train_counts: Dict[str, int] = {}
    test_counts:  Dict[str, int] = {}
    corrupted_train: List[str]   = []
    corrupted_test:  List[str]   = []

    for cls in CLASS_NAMES:
        tr_path = os.path.join(TRAIN_DIR, cls)
        te_path = os.path.join(TEST_DIR,  cls)

        ok_tr, bad_tr = verify_images(tr_path) if os.path.isdir(tr_path) else (0, [])
        ok_te, bad_te = verify_images(te_path) if os.path.isdir(te_path) else (0, [])

        train_counts[cls] = ok_tr
        test_counts[cls]  = ok_te
        corrupted_train   += bad_tr
        corrupted_test    += bad_te

    total_tr = sum(train_counts.values())
    total_te = sum(test_counts.values())

    for cls in CLASS_NAMES:
        tr_pct = 100 * train_counts[cls] / total_tr if total_tr else 0
        te_pct = 100 * test_counts[cls]  / total_te if total_te else 0
        lines.append(
            f"{cls:<14} {train_counts[cls]:>8} {test_counts[cls]:>8} "
            f"{tr_pct:>7.1f}% {te_pct:>7.1f}%"
        )

    lines += [
        "-" * 62,
        f"{'TOTAL':<14} {total_tr:>8} {total_te:>8}",
        "=" * 62,
        f"Corrupted images (train): {len(corrupted_train)}",
        f"Corrupted images (test):  {len(corrupted_test)}",
        "=" * 62,
    ]

    report = "\n".join(lines)
    print(report)

    stats_path = os.path.join(RESULTS_DIR, "dataset_stats.txt")
    with open(stats_path, "w") as f:
        f.write(report + "\n")
    print(f"[[OK]] Dataset stats saved -> {stats_path}\n")

    # Remove corrupted files so they don't block tf.data
    for fp in corrupted_train + corrupted_test:
        try:
            os.remove(fp)
            print(f"  [removed corrupted] {fp}")
        except OSError:
            pass

    return {"train": train_counts, "test": test_counts}


# ==============================================================================
# §2  TF.DATA PIPELINE
# ==============================================================================

def build_dataset(
    data_dir: str,
    input_size: int,
    preprocess_fn,
    is_training: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from *data_dir* using the folder label structure.
    Uses a custom label map so class indices always match FOLDER_MAP.
    """
    # Collect all (path, label) pairs in deterministic order
    image_paths: List[str] = []
    labels:      List[int] = []

    for cls, idx in FOLDER_MAP.items():
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(fpath)
                labels.append(idx)

    image_paths_t = tf.constant(image_paths)
    labels_t      = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths_t, labels_t))

    if is_training:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [input_size, input_size])
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, image_paths, np.array(labels)


# ==============================================================================
# §3  DATA AUGMENTATION
# ==============================================================================

def build_augmentation_layer() -> tf.keras.Sequential:
    """Return a Keras Sequential of augmentation layers (training-only)."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.20),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.20),
    ], name="augmentation")


# ==============================================================================
# §4  MODEL FACTORY
# ==============================================================================

def _classification_head(x, name_prefix: str):
    """Shared classification head for all backbones."""
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Dense(512, activation="relu", name=f"{name_prefix}_dense1")(x)
    x = layers.Dropout(0.4, name=f"{name_prefix}_drop1")(x)
    x = layers.Dense(256, activation="relu", name=f"{name_prefix}_dense2")(x)
    x = layers.Dropout(0.3, name=f"{name_prefix}_drop2")(x)
    x = layers.Dense(N_CLASSES, activation="softmax", name=f"{name_prefix}_out")(x)
    return x


def build_model(model_name: str, input_size: int) -> Tuple[Model, object]:
    """
    Build a functional Keras model for *model_name*.
    Preprocessing is embedded inside the graph.
    Returns (model, preprocess_fn).
    """
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
    from tensorflow.keras.applications import mobilenet_v2, efficientnet, resnet50

    inp = layers.Input(shape=(input_size, input_size, 3), name="input_image")

    # Augmentation applied only during training (training=True flag propagates)
    aug = build_augmentation_layer()
    x   = aug(inp)

    if model_name == "MobileNetV2":
        base  = MobileNetV2(input_shape=(input_size, input_size, 3),
                            include_top=False, weights="imagenet")
        preprocess_fn = mobilenet_v2.preprocess_input
    elif model_name == "EfficientNetB0":
        base  = EfficientNetB0(input_shape=(input_size, input_size, 3),
                               include_top=False, weights="imagenet")
        preprocess_fn = efficientnet.preprocess_input
    elif model_name == "ResNet50":
        base  = ResNet50(input_shape=(input_size, input_size, 3),
                         include_top=False, weights="imagenet")
        preprocess_fn = resnet50.preprocess_input
    else:
        raise ValueError(f"Unknown model: {model_name}")

    base.trainable = False          # Phase 1: frozen
    x = base(x, training=False)    # BN layers stay in inference mode

    outputs = _classification_head(x, model_name.replace(".", "_"))
    model   = Model(inputs=inp, outputs=outputs, name=model_name)
    return model, preprocess_fn


# ==============================================================================
# §5  TWO-PHASE FINE-TUNING
# ==============================================================================

def get_callbacks(ckpt_path: str) -> List:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_model(
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
) -> Tuple[Model, float]:
    """
    Two-phase training.
    Phase 1 — frozen backbone, train head only.
    Phase 2 — unfreeze top-N layers, fine-tune end-to-end.
    Returns (best_model, training_time_seconds).
    """
    cfg        = MODEL_CONFIGS[model_name]
    input_size = cfg["input_size"]
    unfreeze_n = cfg["unfreeze_top"]

    ckpt_path  = os.path.join(MODELS_DIR, f"ckpt_{model_name}.h5")
    t_start    = time.time()

    model, _preprocess_fn = build_model(model_name, input_size)

    # -- Phase 1: Train classification head -----------------------------------
    print(f"\n{'='*62}")
    print(f"  [{model_name}]  Phase 1 — Head Training  (LR=1e-3)")
    print(f"{'='*62}")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=get_callbacks(ckpt_path),
        verbose=1,
    )

    # Reload best Phase-1 weights
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    # -- Phase 2: Unfreeze top-N backbone layers -------------------------------
    print(f"\n{'='*62}")
    print(f"  [{model_name}]  Phase 2 — Fine-Tuning top-{unfreeze_n} layers (LR=1e-5)")
    print(f"{'='*62}")

    # Find backbone layer (first Application layer)
    backbone = None
    for lyr in model.layers:
        if hasattr(lyr, "layers"):           # it's a sub-model (backbone)
            backbone = lyr
            break

    if backbone is not None:
        backbone.trainable = True
        for lyr in backbone.layers[:-unfreeze_n]:
            lyr.trainable = False
        print(f"  Trainable backbone layers: "
              f"{sum(1 for l in backbone.layers if l.trainable)} / {len(backbone.layers)}")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=get_callbacks(ckpt_path),
        verbose=1,
    )

    # Reload best Phase-2 weights
    if os.path.exists(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path)

    elapsed = time.time() - t_start
    return model, elapsed


# ==============================================================================
# §6  EVALUATION
# ==============================================================================

def evaluate_model(
    model:      Model,
    model_name: str,
    test_ds:    tf.data.Dataset,
    y_true:     np.ndarray,
) -> Dict:
    """Full evaluation — returns metrics dict and saves plots."""

    # -- Predict ---------------------------------------------------------------
    y_probs = model.predict(test_ds, verbose=0)          # (N, 4)
    y_pred  = np.argmax(y_probs, axis=1)

    # -- Scalar metrics --------------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    # AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    try:
        auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = 0.0

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(f"\n{'='*62}")
    print(f"  {model_name} — Test Evaluation")
    print(f"{'='*62}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec_m*100:.2f}%  (macro)")
    print(f"  Recall    : {rec_m*100:.2f}%  (macro)")
    print(f"  F1 Score  : {f1_m*100:.2f}%  (macro)")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}\n")

    # -- Confusion Matrix Plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix\nAccuracy: {acc*100:.2f}%", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices", f"{model_name}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[[OK]] Confusion matrix saved -> {cm_path}")

    # -- ROC Curve Plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        cls_auc     = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls}  (AUC = {cls_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"{model_name} — ROC Curves (macro AUC = {auc:.3f})", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curves", f"{model_name}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[[OK]] ROC curves saved -> {roc_path}")

    return {
        "model_name":  model_name,
        "accuracy":    acc,
        "precision":   prec_m,
        "recall":      rec_m,
        "f1":          f1_m,
        "f1_weighted": f1_w,
        "auc":         auc,
        "confusion_matrix": cm,
        "y_probs":     y_probs,
        "y_pred":      y_pred,
    }


# ==============================================================================
# §7  ERROR ANALYSIS
# ==============================================================================

def error_analysis(
    model_name:  str,
    image_paths: List[str],
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    max_per_pair: int = 4,
):
    """Find misclassified samples, print most confused pairs, save image grids."""

    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print(f"[{model_name}] Perfect predictions — no errors to analyse!")
        return

    # Count confusion pairs
    pair_counts: Dict[Tuple[int,int], List[int]] = {}
    for idx in wrong_idx:
        pair = (int(y_true[idx]), int(y_pred[idx]))
        pair_counts.setdefault(pair, []).append(idx)

    print(f"\n[{model_name}] Error Analysis — {len(wrong_idx)} misclassified samples")
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -len(x[1]))
    for (true_c, pred_c), idxs in sorted_pairs[:5]:
        print(f"  {CLASS_NAMES[true_c]:>12} -> {CLASS_NAMES[pred_c]:<12}  "
              f"({len(idxs)} errors)")

    # Visualise top-confused pair
    (top_true, top_pred), top_idxs = sorted_pairs[0]
    sample_idxs = top_idxs[:max_per_pair]

    n_cols = min(len(sample_idxs), max_per_pair)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_idxs):
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#222")
        ax.set_title(
            f"True: {CLASS_NAMES[top_true]}\nPred: {CLASS_NAMES[top_pred]}",
            fontsize=9, color="red",
        )
        ax.axis("off")

    fig.suptitle(
        f"{model_name} — Most Confused: "
        f"{CLASS_NAMES[top_true]} -> {CLASS_NAMES[top_pred]}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    err_path = os.path.join(RESULTS_DIR, "error_analysis", f"{model_name}_errors.png")
    plt.savefig(err_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[[OK]] Error analysis grid saved -> {err_path}")


# ==============================================================================
# §8  STRATIFIED K-FOLD CROSS VALIDATION (on training data, single model)
# ==============================================================================

def cross_validate_model(
    model_name: str,
    all_paths:  List[str],
    all_labels: np.ndarray,
    input_size: int,
    preprocess_fn,
    n_folds: int = CV_FOLDS,
) -> Tuple[float, float]:
    """
    Stratified K-fold CV on the training dataset.
    Returns (mean_accuracy, std_accuracy).
    """
    print(f"\n{'='*62}")
    print(f"  [{model_name}]  {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*62}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\n  -- Fold {fold_idx + 1}/{n_folds} --")

        tr_paths = [all_paths[i] for i in tr_idx]
        va_paths = [all_paths[i] for i in va_idx]
        tr_labels = all_labels[tr_idx]
        va_labels = all_labels[va_idx]

        def make_ds(paths, lbs, training):
            paths_t  = tf.constant(paths)
            labels_t = tf.constant(lbs, dtype=tf.int32)
            ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
            if training:
                ds = ds.shuffle(len(paths), seed=SEED)

            def _load(path, label):
                raw   = tf.io.read_file(path)
                image = tf.image.decode_jpeg(raw, channels=3)
                image = tf.image.resize(image, [input_size, input_size])
                image = tf.cast(image, tf.float32)
                image = preprocess_fn(image)
                return image, label

            return (ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .cache()
                      .prefetch(tf.data.AUTOTUNE))

        tr_ds = make_ds(tr_paths, tr_labels, training=True)
        va_ds = make_ds(va_paths, va_labels, training=False)

        # Light model (Phase 1 only — frozen backbone, head only)
        fold_model, _ = build_model(model_name, input_size)
        fold_model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        fold_model.fit(
            tr_ds, validation_data=va_ds,
            epochs=8,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0,
        )

        va_probs = fold_model.predict(va_ds, verbose=0)
        va_preds = np.argmax(va_probs, axis=1)
        fold_acc = accuracy_score(va_labels, va_preds)
        fold_scores.append(fold_acc)
        print(f"  Fold {fold_idx + 1} accuracy: {fold_acc*100:.2f}%")

        del fold_model
        tf.keras.backend.clear_session()

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    print(f"\n  [{model_name}] CV Result: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    return mean_acc, std_acc


# ==============================================================================
# §9  BENCHMARK TABLE
# ==============================================================================

def print_benchmark_table(results: List[Dict], times: Dict[str, float]):
    header = (
        f"\n{'='*90}\n"
        f"  {'Model':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'AUC':>8} {'Time(s)':>9}\n"
        f"{'-'*90}"
    )
    rows = []
    for r in results:
        t = times.get(r["model_name"], 0)
        rows.append(
            f"  {r['model_name']:<16} "
            f"{r['accuracy']*100:>8.2f}% "
            f"{r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>7.2f}% "
            f"{r['f1']*100:>7.2f}% "
            f"{r['auc']:>8.4f} "
            f"{t:>8.0f}s"
        )
    footer = "=" * 90
    table  = header + "\n" + "\n".join(rows) + "\n" + footer

    print("\n\n" + table)
    bm_path = os.path.join(RESULTS_DIR, "benchmark_table.txt")
    with open(bm_path, "w") as f:
        f.write(table + "\n")
    print(f"[[OK]] Benchmark table saved -> {bm_path}")


# ==============================================================================
# §10  MODEL METADATA  (for Flask auto-detection)
# ==============================================================================

def save_metadata(best_result: Dict, best_model_name: str, input_size: int):
    meta = {
        "model_name":    best_model_name,
        "input_size":    input_size,
        "accuracy":      round(float(best_result["accuracy"]),  4),
        "precision":     round(float(best_result["precision"]), 4),
        "recall":        round(float(best_result["recall"]),    4),
        "f1":            round(float(best_result["f1"]),        4),
        "auc":           round(float(best_result["auc"]),       4),
        "class_mapping": FOLDER_MAP,
        "class_names":   CLASS_NAMES,
        "n_classes":     N_CLASSES,
        "trained_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(META_SAVE_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[[OK]] Model metadata saved -> {META_SAVE_PATH}")
    return meta


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    pipeline_start = time.time()

    # -- §1  Dataset Analysis --------------------------------------------------
    print("\n" + "#" * 62)
    print("  NeuroIntel Training Pipeline 2.0 — Research Grade")
    print("#" * 62)
    stats = analyze_dataset()

    # -- Check GPU -------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n[GPU] Detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n[CPU] No GPU detected — training on CPU (may be slow)")

    # -- Collect training image paths for CV -----------------------------------
    all_train_paths:  List[str] = []
    all_train_labels: List[int] = []
    for cls, idx in FOLDER_MAP.items():
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                all_train_paths.append(fpath)
                all_train_labels.append(idx)
    all_train_labels_arr = np.array(all_train_labels)

    # -- §2  Collect test image paths for error analysis -----------------------
    test_image_paths: List[str] = []
    test_labels_list: List[int] = []
    for cls, idx in FOLDER_MAP.items():
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                test_image_paths.append(fpath)
                test_labels_list.append(idx)
    y_test_arr = np.array(test_labels_list)

    # -- Train & evaluate each model -------------------------------------------
    all_results:  List[Dict]          = []
    all_models:   Dict[str, Model]    = {}
    training_times: Dict[str, float]  = {}
    cv_results:   Dict[str, Tuple]    = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        input_size = cfg["input_size"]
        print(f"\n\n{'#'*62}")
        print(f"  Training: {model_name}  (input: {input_size}×{input_size})")
        print(f"{'#'*62}")

        # Resolve preprocess_fn for dataset building
        from tensorflow.keras.applications import mobilenet_v2, efficientnet, resnet50
        preprocess_map = {
            "MobileNetV2":   mobilenet_v2.preprocess_input,
            "EfficientNetB0": efficientnet.preprocess_input,
            "ResNet50":      resnet50.preprocess_input,
        }
        preprocess_fn = preprocess_map[model_name]

        # Build tf.data pipelines
        train_ds, _, _ = build_dataset(
            TRAIN_DIR, input_size, preprocess_fn, is_training=True
        )
        val_ds, test_paths_ds, y_test_ds = build_dataset(
            TEST_DIR, input_size, preprocess_fn, is_training=False
        )

        # -- Two-phase training ------------------------------------------------
        model, elapsed = train_model(model_name, train_ds, val_ds)
        training_times[model_name] = elapsed

        # -- Evaluate on test set ----------------------------------------------
        result = evaluate_model(model, model_name, val_ds, y_test_arr)
        result["training_time"] = elapsed
        all_results.append(result)
        all_models[model_name] = model

        # -- Error analysis ----------------------------------------------------
        error_analysis(
            model_name,
            test_image_paths,
            y_test_arr,
            result["y_pred"],
        )

        # -- Cross-validation (Phase 1 only for speed) -------------------------
        cv_mean, cv_std = cross_validate_model(
            model_name,
            all_train_paths,
            all_train_labels_arr,
            input_size,
            preprocess_fn,
            n_folds=CV_FOLDS,
        )
        cv_results[model_name] = (cv_mean, cv_std)

        tf.keras.backend.clear_session()

    # -- §9  Research Benchmark Table ------------------------------------------
    print_benchmark_table(all_results, training_times)

    # Cross-validation summary
    print("\n  Cross-Validation Summary (5-Fold, Phase-1 only):")
    print(f"  {'Model':<18} {'Mean Acc':>9} {'Std':>7}")
    print("  " + "-" * 38)
    for mname, (mean, std) in cv_results.items():
        print(f"  {mname:<18} {mean*100:>8.2f}% ±{std*100:.2f}%")

    # -- §11  Best Model Selection ---------------------------------------------
    best_result = max(all_results, key=lambda r: r["accuracy"])
    best_name   = best_result["model_name"]
    best_model  = all_models[best_name]
    best_input  = MODEL_CONFIGS[best_name]["input_size"]

    print(f"\n[*] Best model: {best_name}  "
          f"({best_result['accuracy']*100:.2f}% accuracy)")
    best_model.save(MODEL_SAVE_PATH)
    print(f"[[OK]] Best model saved -> {MODEL_SAVE_PATH}")

    # -- §12  Metadata for Flask dynamic loading -------------------------------
    meta = save_metadata(best_result, best_name, best_input)

    # -- Final Summary ---------------------------------------------------------
    total_time = time.time() - pipeline_start
    print("\n\n" + "#" * 62)
    print("  FINAL RESULTS")
    print("#" * 62)
    print(f"  Best Model      : {best_name}")
    print(f"  Best Accuracy   : {best_result['accuracy']*100:.2f}%")
    print(f"  Precision       : {best_result['precision']*100:.2f}%")
    print(f"  Recall          : {best_result['recall']*100:.2f}%")
    print(f"  F1 Score        : {best_result['f1']*100:.2f}%")
    print(f"  AUC-ROC         : {best_result['auc']:.4f}")
    print(f"  Training Time   : {best_result['training_time']:.0f}s")
    print(f"  Total Pipeline  : {total_time:.0f}s")
    print("#" * 62)

    return meta


if __name__ == "__main__":
    main()
