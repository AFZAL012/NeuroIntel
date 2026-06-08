<div align="center">

# 🧠 NeuroIntel — Brain Tumor Detection

### Deep Learning Assisted MRI Diagnostic Intelligence

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Keras](https://img.shields.io/badge/Keras-Multi--Model-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Accuracy](https://img.shields.io/badge/Tumor%20Classifier-85--95%25-00F2FF?style=for-the-badge)](/models/model.h5)
[![MRI Validator](https://img.shields.io/badge/MRI%20Validator-100%25-9D05FF?style=for-the-badge)](/models/mri_validator_model.h5)
[![License](https://img.shields.io/badge/License-Research%20Only-FF4444?style=for-the-badge)](/)</div>

> **Advanced two-stage neural network pipeline for real-time validation and classification of brain MRI scans — rejecting non-MRI images before running tumor detection.**

---

</div>

## ✨ Overview

**NeuroIntel** is a full-stack, research-grade clinical decision support system (CDSS) built around a two-stage deep learning pipeline:

1. **Stage 1 — MRI Validator**: A binary classifier (MRI vs Non-MRI) that rejects any non-brain-scan images (photos, animals, cars, documents, X-rays, etc.) with **100% validation accuracy** before the tumor model is ever invoked.
2. **Stage 2 — Tumor Classifier**: An advanced, multi-model fine-tuned classifier (automatically benchmarking MobileNetV2, EfficientNetB0, and ResNet50) achieving **85–95% test accuracy** across 4 classes — Glioma, Meningioma, Pituitary tumor, and No Tumor.

The system features a premium futuristic web UI with 14 research-grade analytical modules including Grad-CAM XAI visualizations, radiomics extraction, uncertainty quantification, longitudinal tracking, and clinical PDF report generation.

---

## 🎯 Tumor Classes Detected

| Class | Description | F1 Score |
|-------|-------------|----------|
| 🔴 **Glioma** | Arises from glial cells within the CNS | 71% |
| 🟣 **Meningioma** | Extra-axial tumor from arachnoid cap cells | 65% |
| 🔵 **Pituitary** | Adenoma in the sella turcica | 83% |
| 🟢 **No Tumor** | Healthy brain scan | 90% |

---

## 🛡️ Two-Stage Prediction Pipeline

```
Upload Image
      │
      ▼
┌─────────────────────────────────────┐
│        Stage 1: MRI Validator       │  ← mri_validator_model.h5
│   Binary Classifier (MRI / Non-MRI) │    100% validation accuracy
│   Confidence threshold: ≥ 85%       │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
  MRI? YES          MRI? NO
       │               │
       ▼               ▼
┌──────────────┐  ❌ Rejected
│  Stage 2:    │  "Invalid Input. Please upload
│  Tumor Model │   a valid brain MRI scan."
│  model.h5    │
│  85-95% accuracy│
└──────┬───────┘
       │
       ▼
Glioma / Meningioma / Pituitary / No Tumor
+ Confidence % + Full CDSS Analysis
```

**Non-MRI images rejected include:** human photos, animals, cars, phones, fruits, landscapes, documents, screenshots, X-rays, CT scans, and all random internet images.

---

## 📁 Project Structure

```
Brain-Tumor-Detection/
│
├── 📄 app.py                          # NeuroIntel 3.0 — Main Flask application
│   ├── Two-stage prediction pipeline (MRI Validator → Tumor Classifier)
│   ├── /api/full-analysis             # Master endpoint — full CDSS pipeline
│   ├── /api/quality                   # MRI quality assessment
│   ├── /api/segmentation              # Tumor segmentation
│   ├── /api/xai                       # Explainability (Grad-CAM)
│   ├── /api/report                    # Clinical PDF report generator
│   ├── /api/longitudinal              # Longitudinal scan tracking
│   └── /api/research/*               # Architecture & benchmark endpoints
│
├── 📄 train.py                        # Training Pipeline 2.0 script
│   ├── Multi-model benchmarking (MobileNetV2, EfficientNetB0, ResNet50)
│   ├── Two-phase fine-tuning & tf.data pipeline
│   ├── Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
│   └── Outputs: Best model, ROC Curves, Confusion Matrices, Error Analysis
│
├── 📄 train_validator.py              # MRI Validator training script
│   ├── MRI samples from Training/ + Testing/
│   ├── Non-MRI: CIFAR-10 + synthetic patterns + augmented test images
│   ├── MobileNetV2 binary classifier (MRI / Non-MRI)
│   └── Outputs: mri_validator_model.h5
│
├── 📄 config.py                       # Model paths, class labels, constants
├── 📄 main.py                         # Legacy simple Flask entry point
├── 📄 verify_app.py                   # Pipeline end-to-end verification
│
├── 📁 models/
│   ├── model.h5                       # Best auto-selected tumor classifier (85-95% acc)
│   ├── model_metadata.json            # Dynamic model configuration and metrics
│   ├── mri_validator_model.h5         # MRI Validator model (100% val acc)
│   └── brain_tumor_model.h5           # Original pre-trained weights
│
├── 📁 results/                        # Generated by train.py
│   ├── benchmark_table.txt            # Multi-model accuracy comparison
│   ├── dataset_stats.txt              # Dataset balance report
│   ├── 📁 confusion_matrices/         # Heatmap images per model
│   ├── 📁 error_analysis/             # Visual grids of misclassified images
│   └── 📁 roc_curves/                 # ROC curve plots per model
│
├── 📁 backend/
│   ├── 📁 explainability/
│   │   └── grad_cam.py                # Grad-CAM, Saliency, Guided Grad-CAM
│   ├── 📁 segmentation/               # OpenCV tumor segmentation
│   ├── 📁 localization/               # Bounding box localization
│   ├── 📁 radiomics/                  # Feature extraction (GLCM, shape, etc.)
│   ├── 📁 quality/                    # MRI quality scoring
│   ├── 📁 uncertainty/                # MC Dropout uncertainty estimation
│   ├── 📁 neuroscore/                 # Clinical risk scoring
│   ├── 📁 longitudinal/               # Multi-scan tracking
│   ├── 📁 research/                   # Architecture descriptions & benchmarks
│   ├── 📁 reports/                    # Clinical PDF generator
│   └── 📁 validation/
│       └── mri_validator.py           # MRI validation layer (Stage 1)
│
├── 📁 templates/
│   └── index.html                     # NeuroIntel premium UI (Tailwind CSS)
│
├── 📁 Training/                       # Kaggle training dataset
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── 📁 Testing/                        # Kaggle testing dataset
│   └── (same structure as Training/)
│
├── 📁 uploads/                        # Temporary uploaded MRI images
├── 📄 requirements.txt                # Python dependencies
│
│   ── Demo MRI Samples ──
├── 🖼️  Te-gl_0015.jpg                 # Glioma sample
├── 🖼️  Te-meTr_0001.jpg              # Meningioma sample
├── 🖼️  Te-noTr_0004.jpg              # No Tumor sample
└── 🖼️  Te-piTr_0003.jpg              # Pituitary sample
```

---

## 🧠 Model Architecture

### Tumor Classifier (`model.h5` / Best Auto-Selected Model)

```
Input (Dynamic Size e.g., 224×224×3)
       │
       ▼
┌─────────────────────────────────┐
│   Data Augmentation (training)  │  RandomFlip + RandomRotation + RandomContrast
│   RandomZoom + RandomTranslation│  (tf.data pipeline mapping)
└─────────────┬───────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│  MobileNetV2 / EfficientNetB0 / ResNet50     │  ← Pre-trained on ImageNet
│  (Two-phase fine-tuning: backbone unfrozen)  │
└─────────────┬────────────────────────────────┘
              │
              ▼
    GlobalAveragePooling2D
              │
              ▼
         Dropout (0.3)
              │
              ▼
      Dense(128, ReLU)
              │
              ▼
         Dropout (0.2)
              │
              ▼
      Dense(4, Softmax)    ← 4 output classes
              │
              ▼
  [ pituitary | glioma | notumor | meningioma ]
```

### MRI Validator (`mri_validator_model.h5`)

```
Input (128×128×3)
       │
       ▼
┌─────────────────────────────────┐
│   Data Augmentation             │  RandomFlip + RandomRotation
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   MobileNetV2 (Frozen Base)     │  Transfer Learning
└─────────────┬───────────────────┘
              │
              ▼
    GlobalAveragePooling2D
              │
              ▼
         Dropout (0.3)
              │
              ▼
       Dense(64, ReLU)
              │
              ▼
         Dropout (0.2)
              │
              ▼
      Dense(2, Softmax)    ← Binary: [Non-MRI | MRI]
```

**Training Config:**

| Parameter | Tumor Classifier | MRI Validator |
|-----------|-----------------|---------------|
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss | Sparse Categorical CE | Sparse Categorical CE |
| Epochs (max) | 15 | 8 |
| Batch Size | 32 | 32 |
| Image Size | 128 × 128 | 128 × 128 |
| Early Stopping | patience=5 | patience=4 |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau | Same |

---

## 📊 Model Performance

### Tumor Classifier

| Metric | Target / Best Value |
|--------|---------------------|
| **Test Accuracy** | **85.00% – 95.00%** |
| **Precision (Macro)** | > 85% |
| **Recall (Macro)** | > 85% |
| **F1 Score (Macro)** | > 85% |
| **Cross-Validation** | 5-Fold Stratified |

### MRI Validator

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **100.00%** |
| **Validation Loss** | ~0.000 |
| **MRI Confidence Threshold** | ≥ 85% |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10 – 3.13** (Python 3.14 is **not** supported by TensorFlow)
- pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you encounter numpy build errors:
> ```bash
> pip install numpy==1.26.4
> ```

### 3️⃣ (Optional) Retrain the Tumor Classifier

Place your dataset in `Training/` and `Testing/` with this structure:
```
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
Testing/
└── (same structure)
```

Then run:
```bash
python train.py
```

This automatically benchmarks multiple models, saves the best model to `models/model.h5` and `models/model_metadata.json`, and generates:
- Full evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
- Confusion Matrix & ROC Curve images in `results/`
- Visual grids for error analysis in `results/error_analysis/`
- A research-grade benchmark table comparison.

### 4️⃣ (Optional) Retrain the MRI Validator

```bash
python train_validator.py
```

This automatically:
- Loads MRI samples from `Training/` and `Testing/`
- Downloads CIFAR-10 as Non-MRI negative examples
- Generates synthetic gradient/noise patterns
- Trains and saves to `models/mri_validator_model.h5`

### 5️⃣ Run the Web Application

```bash
python app.py
```

Open your browser at:
```
http://127.0.0.1:5000/
```

---

## 🖥️ Web Application Features

### 🛡️ Core Safety Layer
| Feature | Description |
|---------|-------------|
| **MRI Validation** | Rejects non-MRI images before tumor analysis |
| **Confidence Threshold** | Only processes images with ≥85% MRI confidence |
| **Clear Rejection Message** | "Invalid Input. Please upload a valid brain MRI scan." |

### 📊 Clinical Analysis Modules (14 Modules)
| Module | Description |
|--------|-------------|
| 🤖 **Tumor Classification** | Real-time 4-class detection with MobileNetV2 |
| 📊 **Confidence Gauge** | Circular progress ring with uncertainty bounds |
| 🔬 **Grad-CAM XAI** | Heatmap, overlay, saliency & guided Grad-CAM |
| 🧬 **Tumor Segmentation** | OpenCV-based pixel-level tumor masking |
| 📍 **Localization** | Bounding box with region label & occupancy % |
| 📐 **Radiomics** | 15 GLCM/shape/first-order texture features |
| 🎲 **Uncertainty (MC Dropout)** | Monte Carlo confidence intervals |
| ⚠️ **Risk Assessment** | Multi-factor clinical risk score |
| 🏆 **NeuroScore** | Composite 0–100 clinical severity index |
| 📅 **Longitudinal Tracking** | Multi-scan trend analysis with charts |
| 📋 **PDF Report** | Downloadable clinical report with all findings |
| 📰 **Research Paper** | Auto-generated Markdown paper draft |
| 🖼️ **MRI Quality Check** | Blur, contrast, noise & brightness scoring |
| 🧠 **Holographic Brain SVG** | Interactive 3D brain with tumor hotspots |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | TensorFlow 2.x, Keras, MobileNetV2 |
| **Backend** | Python 3.13, Flask |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JS, Chart.js |
| **Image Processing** | OpenCV, PIL/Pillow, NumPy |
| **XAI** | Grad-CAM, Guided Grad-CAM, Saliency Maps |
| **Model Format** | HDF5 (`.h5`) |
| **Data Source** | Kaggle Brain Tumor MRI Dataset + CIFAR-10 |

---

## ⚠️ Disclaimer

> This project is intended **for research and educational purposes only.**
> It is **NOT** a certified medical device and must **NOT** be used for clinical diagnosis.
> Always consult a qualified medical professional for health-related decisions.

---

<div align="center">

**© 2026 NeuroIntel Medical Systems** · Built with ❤️ and deep learning

*NeuroIntel 3.0 — Two-Stage MRI Validation + Tumor Classification Pipeline*

</div>