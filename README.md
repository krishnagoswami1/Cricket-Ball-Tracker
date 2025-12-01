# Cricket Ball Detection Project

End-to-End Dataset Creation, Preprocessing, YOLO Training, and Inference

This repository contains the complete, reproducible workflow for building a **cricket ball detection** system using **Ultralytics YOLO**, **Roboflow annotations**, and **Google Colab GPU acceleration**. The project spans raw-video preprocessing, dataset creation, model training, and final prediction on cricket videos.

---

## 1. Project Overview

The objective of this project is to build a robust cricket-ball detector capable of localizing the ball across diverse match conditions. The workflow includes:

1. **Baseline preprocessing** of raw cricket videos
2. **Annotation generation using Roboflow**
3. **Dataset export in YOLO format**
4. **YOLO-based model training on Google Colab**
5. **Inference & prediction pipeline for new videos**

This repository is structured to provide a clean, modular pipeline suitable for research, demonstrations, and production use.

---

## 2. Tools and Technologies

### Core Libraries and Platforms

| Component            | Usage                                                     |
| -------------------- | --------------------------------------------------------- |
| **Roboflow**         | Dataset annotation, augmentation, and YOLO dataset export |
| **OpenCV (cv2)**     | Video loading, resizing, FPS normalization                |
| **Ultralytics YOLO** | Training & inference                                      |
| **Google Colab GPU** | Accelerated training environment                          |
| **TensorBoard**      | Training metrics visualization                            |
| **Python 3.x**       | Runtime environment                                       |

---

## 3. Dataset Creation (Annotations via Roboflow)

All cricket-ball annotations were created using **Roboflow**, which provides an efficient interface for:

* Uploading raw video frames
* Drawing bounding boxes around the cricket ball
* Managing dataset versions
* Exporting directly in **YOLOv8 format**

Dataset steps:

1. Uploaded extracted frames into Roboflow
2. Annotated cricket ball in each image
3. Applied optional preprocessing (resizing, augmentations)
4. Exported dataset with Roboflow’s YOLO-ready structure
5. Downloaded dataset programmatically inside Colab using the Roboflow API

---

## 4. Baseline Video Preprocessing

Raw cricket videos often vary in:

* Resolution
* Frame rate
* Codec
* Aspect ratio

A preprocessing notebook normalizes all these attributes to produce uniform training-ready videos.

### Preprocessing operations include:

* Opening videos using `cv2.VideoCapture`
* Standardizing resolution (e.g., 1920×1080)
* Standardizing FPS (e.g., 40 FPS)
* Rewriting via `cv2.VideoWriter`
* Saving outputs to `processed_videos/`

This ensures consistent frame extraction and reduces YOLO training issues related to inconsistent input dimensions.

---

## 5. Training Pipeline (Google Colab)

Training was performed in **Google Colab** due to its accessible GPU environment.

### Important Note About GPU RAM

Colab provided **15 GB GPU RAM**, which caused compatibility constraints.
As a result:

* Larger YOLO models (L or X) could not be used
* The project used **YOLO-M** (or S), which fits within Colab's memory
* Batch sizes were tuned to prevent out-of-memory (OOM) issues
* CUDA cache was cleared periodically using:

```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

### Training script components:

* Load YOLO model:

  ```python
  model = YOLO("yolov8m.pt")
  ```
* Train:

  ```python
  model.train(data="data.yaml", imgsz=640, epochs=50, batch=8)
  ```
* Enable TensorBoard logging
* Save and export results to Google Drive

---

## 6. Prediction / Inference Pipeline

Once the model is trained, the prediction notebook performs:

1. Loading the trained `best.pt` weights
2. Opening the input video via `cv2.VideoCapture`
3. Running YOLO inference frame-by-frame
4. Drawing bounding boxes & confidence scores
5. Displaying the output live
6. Optionally writing annotated video using `cv2.VideoWriter`

This pipeline is optimized for smooth visualization and reusable export.

---

## 7. Repository Structure

```
📂 cricket-ball-detection
 ┣ 📂 data
 │  ┣ 📂 raw_videos
 │  ┣ 📂 processed_videos
 │  ┗ data.yaml
 ┣ 📂 notebooks
 │  ┣ video_preprocessing.ipynb
 │  ┣ model_training.ipynb
 │  ┗ prediction.ipynb
 ┣ 📂 runs
 │  ┗ trained YOLO runs (auto-generated)
 ┣ README.md
 ┗ requirements.txt
```

---

## 8. End-to-End Flow Summary

Below is the complete flow of operations in this project.

### Step 1 — Raw Videos

Start with unprocessed cricket match videos.

### Step 2 — Preprocessing

Normalize FPS, dimensions, and encoding.

### Step 3 — Frame Extraction & Upload

Extract frames → Upload to Roboflow.

### Step 4 — Annotation & Dataset Export

Annotate cricket ball → Export as YOLO dataset.

### Step 5 — Training in Colab

Train YOLO model using GPU.

### Step 6 — Evaluate & Log

Use TensorBoard to monitor loss, mAP, precision, recall.

### Step 7 — Prediction

Run inference on match videos and save results.

---

## 9. Key Challenges Encountered

### 1. GPU Memory Limitations (15 GB on Colab)

* Had to avoid large YOLO models
* Tuned batch sizes
* Included manual CUDA cache clearing

### 2. Annotation Consistency

Ensured every image had a matching label file using mismatch-check logic.

### 3. Preprocessing Variability

Standardized all videos before training to avoid mismatched frame sizes.

---

## 10. Results

The trained YOLO model successfully detects cricket balls with high accuracy, even under challenging conditions like:

* Motion blur
* Varying frame resolutions
* Outdoor lighting variations
* Fast ball movement

Outputs can be used for:
Trajectory analysis, highlight extraction, analytics dashboards, or downstream classifiers.

---

## 11. How to Reproduce

1. Clone the repo
2. Install dependencies
3. Run preprocessing notebook
4. Upload processed frames to Roboflow
5. Download YOLO dataset
6. Train using Colab GPU
7. Run prediction notebook

---

## 12. Future Improvements

* Multi-camera ball tracking
* 3D trajectory estimation
* Deployment as a web-service (FastAPI / Streamlit)
* Integration with cricket analytics tools



Just tell me.
