import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def verify_pipeline():
    model_path = 'models/model.h5'
    
    print("1. Verifying model file existence...")
    if not os.path.exists(model_path):
        print(f"FAILED: Model file not found at {model_path}")
        return False
    print("SUCCESS: Model file found.")
    
    print("\n2. Loading model...")
    try:
        model = load_model(model_path)
        print("SUCCESS: Model loaded successfully.")
    except Exception as e:
        print(f"FAILED: Error loading model: {e}")
        return False
        
    print("\n3. Verifying model architecture...")
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    
    if input_shape[1:3] != (128, 128):
        print(f"FAILED: Expected input size 128x128, but got {input_shape[1:3]}")
        return False
    if output_shape[1] != 4:
        print(f"FAILED: Expected output classes to be 4, but got {output_shape[1]}")
        return False
    print("SUCCESS: Model shapes are correct.")
    
    print("\n4. Running inference on sample MRI images...")
    class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
    sample_images = [
        ("Te-gl_0015.jpg", "glioma"),
        ("Te-meTr_0001.jpg", "meningioma"),
        ("Te-noTr_0004.jpg", "notumor"),
        ("Te-piTr_0003.jpg", "pituitary")
    ]
    
    success = True
    for img_name, true_label in sample_images:
        if not os.path.exists(img_name):
            print(f"WARNING: Sample image {img_name} not found in workspace.")
            continue
            
        try:
            # Preprocess image exactly as done in app
            img = load_img(img_name, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds, axis=1)[0]
            confidence = np.max(preds, axis=1)[0]
            pred_label = class_labels[pred_idx]
            
            print(f"Image: {img_name} | True label: {true_label} -> Predicted: {pred_label} (Confidence: {confidence * 100:.2f}%)")
        except Exception as e:
            print(f"FAILED: Inference failed for {img_name}: {e}")
            success = False
            
    if success:
        print("\nSUCCESS: All pipeline verifications completed successfully!")
        return True
    else:
        print("\nFAILED: Some pipeline verifications failed.")
        return False

if __name__ == '__main__':
    verify_pipeline()
