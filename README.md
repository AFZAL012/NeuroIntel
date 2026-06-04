<div align="center">

# 🧠 NeuroIntel — Brain Tumor Detection

### Deep Learning Assisted MRI Diagnostic Intelligence

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-86.5%25-00F2FF?style=for-the-badge)](/)
[![License](https://img.shields.io/badge/License-Research%20Only-9D05FF?style=for-the-badge)](/)

> **Advanced neural network analysis for real-time identification and classification of Glioma, Meningioma, and Pituitary tumors from brain MRI scans.**

---

</div>

## ✨ Overview

**NeuroIntel** is a computer vision–powered web application that uses a fine-tuned **MobileNetV2** deep learning model to detect and classify brain tumors from MRI images. It achieves **86.5% test accuracy** across 4 classes — Glioma, Meningioma, Pituitary tumor, and No Tumor — with an average inference time of just **~82ms**.

The project features a premium, futuristic web UI built with Flask and Tailwind CSS, complete with an interactive holographic brain SVG, real-time confidence gauges, and clinical pathology references.

---

## 🎯 Tumor Classes Detected

| Class | Description | Characteristics |
|-------|-------------|-----------------|
| 🔴 **Glioma** | Arises from glial cells within the CNS | Highly infiltrative, challenging resection |
| 🟣 **Meningioma** | Extra-axial tumor from arachnoid cap cells | Typically benign, causes focal deficits |
| 🔵 **Pituitary** | Adenoma in the sella turcica | Causes visual & endocrine dysfunction |
| 🟢 **No Tumor** | Healthy brain scan | Normal ventricular volume & symmetry |

---

## 🗺️ Project Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT FLOW                             │
└─────────────────────────────────────────────────────────────────┘

  📁 Dataset                    🧠 Model Training               🌐 Web App
  ─────────                    ──────────────────               ──────────
  MRI Images          ──►      MobileNetV2 (pretrained)   ──►  Flask Server
  (128×128 RGB)                + Custom Head                   (main.py)
                               [train.py]                           │
  4 Categories:                     │                               │
  • glioma_tumor                    ▼                      ┌────────▼────────┐
  • meningioma_tumor          models/model.h5              │  User Uploads   │
  • pituitary_tumor            (saved weights)             │   MRI Image     │
  • no_tumor                        │                      └────────┬────────┘
                                    │                               │
                                    │                    ┌──────────▼──────────┐
                                    └───────────────►    │  predict_tumor()    │
                                                         │  • Load & resize    │
                                                         │  • preprocess_input │
                                                         │  • model.predict()  │
                                                         │  • argmax + label   │
                                                         └──────────┬──────────┘
                                                                    │
                                                         ┌──────────▼──────────┐
                                                         │   Result + Conf %   │
                                                         │  Rendered in UI     │
                                                         │  (index.html)       │
                                                         └─────────────────────┘
```

---

## 📁 Project Structure

```
Brain-Tumor-Detection/
│
├── 📄 main.py                          # Flask web application entry point
│   ├── Flask routes (GET / POST)
│   ├── predict_tumor() function
│   └── File upload & serving logic
│
├── 📄 train.py                         # Model training script
│   ├── Data loading & preprocessing
│   ├── MobileNetV2 transfer learning
│   └── Model evaluation & saving
│
├── 📄 verify_app.py                    # Pipeline verification script
│   ├── Model loading check
│   └── End-to-end inference test
│
├── 📄 notebook_code.py                 # Exported Jupyter notebook code
│
├── 📓 brain_tumour_detection_using_deep_learning.ipynb
│   └── Full exploratory notebook with EDA + training
│
├── 📄 requirements.txt                 # Python dependencies
│
├── 📁 models/
│   └── model.h5                        # Trained MobileNetV2 weights
│
├── 📁 templates/
│   └── index.html                      # NeuroIntel premium UI (Tailwind + SVG)
│
├── 📁 uploads/                         # Temporary uploaded MRI images
│
│   ── Demo MRI Samples ──
├── 🖼️  Te-gl_0015.jpg                  # Glioma sample
├── 🖼️  Te-meTr_0001.jpg               # Meningioma sample
├── 🖼️  Te-noTr_0004.jpg               # No Tumor sample
└── 🖼️  Te-piTr_0003.jpg               # Pituitary sample
```

---

## 🧠 Model Architecture

```
Input (128×128×3)
       │
       ▼
┌─────────────────────────────┐
│   MobileNetV2 (Frozen)      │  ← Pre-trained on ImageNet
│   1,280 base feature maps   │    Transfer Learning
└─────────────┬───────────────┘
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
      Dense(4, Softmax)         ← 4 output classes
              │
              ▼
    [ pituitary | glioma | notumor | meningioma ]
```

**Training Config:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 4 |
| Batch Size | 32 |
| Image Size | 128 × 128 |
| Base Model | MobileNetV2 (ImageNet weights) |

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

> ⚠️ If you encounter numpy build errors, install a pre-built wheel:
> ```bash
> pip install numpy==1.26.4
> ```

### 3️⃣ (Optional) Retrain the Model

If you want to retrain from scratch, place your dataset at `temp_dataset/` with the structure:
```
temp_dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── Testing/
    └── (same structure)
```

Then run:
```bash
python train.py
```

### 4️⃣ Verify the Pipeline

```bash
python verify_app.py
```

Expected output:
```
✅ SUCCESS: All pipeline verifications completed successfully!
```

### 5️⃣ Run the Web Application

```bash
python main.py
```

Open your browser at:
```
http://127.0.0.1:5000/
```

---

## 🖥️ Web Application Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Drag & Drop Upload** | Upload any JPG/PNG/WEBP MRI scan |
| 🤖 **AI Classification** | Real-time tumor detection in ~82ms |
| 📊 **Confidence Gauge** | Circular progress ring showing prediction confidence |
| 🧬 **Holographic Brain SVG** | Interactive 3D brain visualization with tumor highlighting |
| 📋 **Demo Samples** | One-click Glioma, Meningioma, Pituitary & Healthy demos |
| 📈 **Model Analytics** | Confusion matrix & training loss curves |
| 🏥 **Clinical Education** | Pathology references for each tumor type |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.5% |
| **Inference Time** | ~82ms |
| **Total Scans Mapped** | 3,000+ |
| **Pituitary Recall** | 100% (74/74) |
| **Glioma Recall** | ~91% (91/100) |
| **Meningioma Recall** | ~94% (110/117) |
| **No Tumor Recall** | ~98% (103/105) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | TensorFlow 2.20, Keras, MobileNetV2 |
| **Backend** | Python 3.13, Flask |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JS |
| **Image Processing** | PIL/Pillow, NumPy |
| **Visualization** | SVG animations, CSS glassmorphism |
| **Model Format** | HDF5 (`.h5`) |

---

## ⚠️ Disclaimer

> This project is intended **for research and educational purposes only**.  
> It is **NOT** a certified medical device and must **NOT** be used for clinical diagnosis.  
> Always consult a qualified medical professional for health-related decisions.

---

<div align="center">

**© 2026 NeuroIntel Medical Systems** · Built with ❤️ and deep learning

</div>