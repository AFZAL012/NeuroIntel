"""
MRI Validation Layer using MobileNetV2 Binary Classifier.
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import Dict, Any

_validator_model = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "mri_validator_model.h5")

def validate_mri(image_path: str) -> Dict[str, Any]:
    global _validator_model
    
    # Load model if not already loaded
    if _validator_model is None:
        if not os.path.exists(MODEL_PATH):
            # Safe fallback if model file does not exist yet (during initial startup or test setups)
            print(f"Validator model not found at {MODEL_PATH}, using fallback heuristics.")
            return _validate_mri_heuristics(image_path)
        try:
            _validator_model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading validator model: {e}")
            return {"is_mri": False, "confidence": 0.0}
            
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_arr = np.array(img, dtype=np.float32)
        img_pp = preprocess_input(img_arr)
        img_batch = np.expand_dims(img_pp, axis=0)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return {"is_mri": False, "confidence": 0.0}
        
    try:
        preds = _validator_model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds, axis=1)[0])
        
        # Class 1 = MRI, Class 0 = Non_MRI
        is_mri = (pred_class == 1) and (confidence >= 0.85)
        
        return {
            "is_mri": is_mri,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Inference error in validator: {e}")
        return {"is_mri": False, "confidence": 0.0}

def _validate_mri_heuristics(image_path: str) -> Dict[str, Any]:
    # Fallback to the CV corner checks if the model is not found
    img = cv2.imread(image_path)
    if img is None:
        return {"is_mri": False, "confidence": 0.0}
    h, w = img.shape[:2]
    
    # Color check
    b, g, r = cv2.split(img)
    diff = (np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) + 
            np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))) + 
            np.mean(np.abs(b.astype(np.float32) - r.astype(np.float32)))) / 3.0
    
    grayscale_score = 1.0 if diff < 1.0 else (0.0 if diff > 10.0 else float(1.0 - (diff - 1.0) / 9.0))
    
    # Corner check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    s = int(min(w, h) * 0.12)
    if s < 5: s = 5
    avg_corner = float((np.mean(gray[0:s, 0:s]) + np.mean(gray[0:s, w-s:w]) + 
                        np.mean(gray[h-s:h, 0:s]) + np.mean(gray[h-s:h, w-s:w])) / 4.0)
                        
    corner_score = 1.0 if avg_corner < 10.0 else (0.0 if avg_corner > 30.0 else float(1.0 - (avg_corner - 10.0) / 20.0))
    
    total = (grayscale_score * 0.4) + (corner_score * 0.6)
    return {
        "is_mri": total >= 0.70,
        "confidence": float(round(total, 3))
    }
