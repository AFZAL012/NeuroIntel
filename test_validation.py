import os
import sys
import cv2
import numpy as np
from backend.validation.mri_validator import validate_mri

# We will temporarily print internal scoring values for debug purposes
def debug_mri(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
    h, w = img.shape[:2]
    
    # 1. Grayscale Check
    if len(img.shape) < 3:
        grayscale_score = 1.0
        mean_ch_diff = 0.0
    else:
        b, g, r = cv2.split(img)
        diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        diff_br = np.mean(np.abs(b.astype(np.float32) - r.astype(np.float32)))
        mean_ch_diff = (diff_rg + diff_gb + diff_br) / 3.0
        if mean_ch_diff < 1.0:
            grayscale_score = 1.0
        elif mean_ch_diff > 10.0:
            grayscale_score = 0.0
        else:
            grayscale_score = float(1.0 - (mean_ch_diff - 1.0) / 9.0)

    # 2. Dark Border Check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    border_w = int(w * 0.08)
    border_h = int(h * 0.08)
    border_mask = np.ones((h, w), dtype=np.uint8) * 255
    border_mask[border_h:h-border_h, border_w:w-border_w] = 0
    mean_border_val = float(np.mean(gray[border_mask == 255]))
    if mean_border_val < 15.0:
        border_score = 1.0
    elif mean_border_val > 45.0:
        border_score = 0.0
    else:
        border_score = float(1.0 - (mean_border_val - 15.0) / 30.0)

    # 3. Contour Check
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score_center = 0.0
    score_size = 0.0
    score_border_touch = 0.0
    area_ratio = 0.0
    dist_from_center = 0.0
    touches = 0
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, bw, bh = cv2.boundingRect(largest)
        area_ratio = area / (w * h)
        if 0.15 <= area_ratio <= 0.80:
            score_size = 1.0
        elif area_ratio < 0.05 or area_ratio > 0.95:
            score_size = 0.0
        else:
            if area_ratio < 0.15:
                score_size = float(area_ratio / 0.15)
            else:
                score_size = float((1.0 - area_ratio) / 0.20)
                
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + bw // 2
            cy = y + bh // 2
            
        dist_from_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2) / min(w, h)
        if dist_from_center < 0.12:
            score_center = 1.0
        elif dist_from_center > 0.25:
            score_center = 0.0
        else:
            score_center = float(1.0 - (dist_from_center - 0.12) / 0.13)
            
        if x <= 2: touches += 1
        if y <= 2: touches += 1
        if x + bw >= w - 3: touches += 1
        if y + bh >= h - 3: touches += 1
        
        if touches == 0:
            score_border_touch = 1.0
        elif touches == 1:
            score_border_touch = 0.7
        elif touches == 2:
            score_border_touch = 0.4
        else:
            score_border_touch = 0.0

    contour_score = (score_center * 0.4) + (score_size * 0.4) + (score_border_touch * 0.2)
    total_score = (grayscale_score * 0.35) + (border_score * 0.35) + (contour_score * 0.3)
    
    print(f"DEBUG {image_path}:")
    print(f"  Grayscale: score={grayscale_score:.3f} (diff={mean_ch_diff:.3f})")
    print(f"  Border   : score={border_score:.3f} (mean_border_val={mean_border_val:.3f})")
    print(f"  Contours : score={contour_score:.3f}")
    print(f"    - Center: score={score_center:.3f} (dist={dist_from_center:.3f})")
    print(f"    - Size  : score={score_size:.3f} (ratio={area_ratio:.3f})")
    print(f"    - Touch : score={score_border_touch:.3f} (touches={touches})")
    print(f"  TOTAL    : {total_score:.3f}")

def run_tests():
    non_mri_images = [
        "test_human_face.jpg",
        "test_cat.jpg",
        "test_car.jpg",
        "test_walnut.jpg",
        "test_landscape.jpg"
    ]
    
    mri_images = [
        "Te-gl_0015.jpg",
        "Te-meTr_0001.jpg",
        "Te-noTr_0004.jpg",
        "Te-piTr_0003.jpg"
    ]
    
    print("=" * 60)
    print("RUNNING MRI VALIDATOR TESTS")
    print("=" * 60)
    
    failed = False
    
    print("\n--- Testing Non-MRI Images (Should be rejected) ---")
    for img in non_mri_images:
        if not os.path.exists(img):
            print(f"[FAIL] {img} not found!")
            failed = True
            continue
            
        res = validate_mri(img)
        is_mri = res["is_mri"]
        conf = res["confidence"]
        accepted = is_mri and (conf >= 0.85)
        
        status = "REJECTED (Pass)" if not accepted else "ACCEPTED (FAIL)"
        print(f"Image: {img:<25} | is_mri: {str(is_mri):<5} | confidence: {conf:.3f} | Status: {status}")
        
        if accepted:
            print(f"  [FAIL] Error: {img} was accepted as a valid MRI!")
            debug_mri(img)
            failed = True
            
    print("\n--- Testing Valid MRI Scans (Should be accepted) ---")
    for img in mri_images:
        if not os.path.exists(img):
            print(f"[FAIL] {img} not found!")
            failed = True
            continue
            
        res = validate_mri(img)
        is_mri = res["is_mri"]
        conf = res["confidence"]
        accepted = is_mri and (conf >= 0.85)
        
        status = "ACCEPTED (Pass)" if accepted else "REJECTED (FAIL)"
        print(f"Image: {img:<25} | is_mri: {str(is_mri):<5} | confidence: {conf:.3f} | Status: {status}")
        
        if not accepted:
            print(f"  [FAIL] Error: Valid MRI {img} was rejected! (is_mri: {is_mri}, confidence: {conf})")
            debug_mri(img)
            failed = True
            
    print("=" * 60)
    if failed:
        print("RESULT: [FAIL] SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("RESULT: [SUCCESS] ALL TESTS PASSED")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
