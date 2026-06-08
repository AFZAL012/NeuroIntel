"""
Phase F — Monte Carlo Dropout Uncertainty Estimation
Runs N stochastic forward passes with dropout enabled at inference time
to estimate prediction uncertainty.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List


def estimate_uncertainty(
    model,
    img_array: np.ndarray,
    class_labels: List[str],
    n_iter: int = 20,
) -> Dict[str, Any]:
    """
    Perform Monte Carlo Dropout inference.

    Parameters
    ----------
    model       : loaded Keras model with Dropout layers
    img_array   : preprocessed image array shape (1, H, W, C)
    class_labels: list of class names matching model output order
    n_iter      : number of stochastic forward passes

    Returns
    -------
    dict with keys:
        mean_probs          : list[float]  — mean probability per class
        std_probs           : list[float]  — std per class (uncertainty)
        predicted_class     : str
        predicted_index     : int
        confidence_pct      : float
        uncertainty_pct     : float  — ±% for predicted class
        entropy             : float  — predictive entropy (bits)
        mutual_information  : float  — epistemic uncertainty
        class_results       : list[dict]  — per-class breakdown
    """
    import tensorflow as tf

    # Collect stochastic predictions (learning_phase=1 keeps dropout active)
    predictions = []
    for _ in range(n_iter):
        # tf.keras models: use training=True to keep dropout alive
        preds = model(img_array, training=True)
        predictions.append(preds.numpy()[0])

    preds_arr = np.array(predictions)          # (n_iter, n_classes)
    mean_p    = preds_arr.mean(axis=0)         # (n_classes,)
    std_p     = preds_arr.std(axis=0)          # (n_classes,)

    pred_idx  = int(np.argmax(mean_p))
    pred_cls  = class_labels[pred_idx]
    conf_pct  = float(mean_p[pred_idx]) * 100
    unc_pct   = float(std_p[pred_idx])  * 100

    # Predictive entropy H = -Σ p̄ log(p̄)
    mean_p_safe = np.clip(mean_p, 1e-10, 1.0)
    entropy     = float(-np.sum(mean_p_safe * np.log(mean_p_safe)))

    # Mutual information (epistemic uncertainty)
    # MI = H(ȳ) - (1/T) Σ H(y^(t))
    per_pass_entropy = -np.sum(
        np.clip(preds_arr, 1e-10, 1.0) * np.log(np.clip(preds_arr, 1e-10, 1.0)),
        axis=1
    )
    mi = float(entropy - np.mean(per_pass_entropy))

    class_results = [
        {
            "label":     class_labels[i],
            "mean_prob": round(float(mean_p[i]) * 100, 2),
            "std_prob":  round(float(std_p[i])  * 100, 2),
        }
        for i in range(len(class_labels))
    ]

    return {
        "mean_probs":         [round(float(p) * 100, 2) for p in mean_p],
        "std_probs":          [round(float(s) * 100, 2) for s in std_p],
        "predicted_class":    pred_cls,
        "predicted_index":    pred_idx,
        "confidence_pct":     round(conf_pct, 2),
        "uncertainty_pct":    round(unc_pct, 2),
        "entropy":            round(entropy, 4),
        "mutual_information": round(mi, 4),
        "class_results":      class_results,
    }


def add_mc_dropout_to_model(model):
    """
    For models trained without dropout, inject MC Dropout by wrapping the model.
    Returns a new functional model with a Dropout layer before the output.

    Note: This is a lightweight workaround. For best results, retrain with dropout.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        inputs   = model.input
        features = model.layers[-2].output  # penultimate layer
        dropped  = layers.Dropout(0.2)(features)
        outputs  = model.layers[-1](dropped)
        mc_model = Model(inputs=inputs, outputs=outputs, name=model.name + "_MC")
        return mc_model
    except Exception:
        return model   # return original if patching fails
