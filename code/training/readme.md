# Cricket Ball Detection – Model Training Pipeline

This repository contains the complete workflow for building a cricket-ball detection model using YOLO. The project integrates dataset creation, preprocessing, verification, and training using Google Colab and Roboflow.

---

## 1. Overview

The primary objective of this project is to train a YOLO-based object detection model capable of accurately detecting a cricket ball in video frames. The complete workflow includes:

* Dataset creation and annotation via **Roboflow**
* Video preprocessing and standardization
* Directory and annotation validation
* YOLO training (Ultralytics)
* Logging training metrics using TensorBoard
* Exporting trained model artifacts

---

## 2. Dataset Creation (Using Roboflow)

Annotations for this project were generated using **Roboflow**, an online platform for dataset labeling and management.

### Why Roboflow?

* Web-friendly annotation interface
* Automatic dataset versioning
* Preprocessing utilities
* Export support for YOLO formats

### Dataset Steps

1. Uploaded raw cricket-ball video frames to Roboflow
2. Annotated each frame with bounding boxes for the cricket ball
3. Exported dataset in **YOLOv8 format**
4. Downloaded programmatically using the Roboflow API in Colab

---

## 3. Environment & Hardware

The project was executed on **Google Colab** to leverage GPU acceleration.

### GPU Limitation Note

Colab provided **15 GB GPU RAM**, which caused compatibility challenges with certain YOLO models (e.g., YOLOv8-L, YOLOv8-X).
Therefore:

* A lighter model variant such as **YOLOv8-M** or **YOLOv8-S** was chosen
* Batch sizes were adjusted to fit memory constraints
* CUDA memory was periodically cleared to prevent OOM errors

---

## 4. Notebook Workflow Summary

### 4.1 Installing Dependencies

The notebook installs:

* OpenCV for video processing
* Roboflow for dataset download
* Ultralytics for YOLO training
* TensorBoard for logs

### 4.2 Preprocessing Videos

Raw videos were standardized by:

* Fixing resolution
* Fixing FPS
* Ensuring consistent formatting for downstream training

This ensures all extracted frames match the expected YOLO preprocessing format.

### 4.3 Dataset Verification

Before training:

* Directory structure was inspected
* File existence was validated
* A custom mismatch-check function ensured label-file alignment

### 4.4 Model Training

Training is performed using:

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="data.yaml",
    imgsz=640,
    epochs=50,
    batch=8,
    device=0
)
```

TensorBoard logging is enabled for visualization of:

* Loss curves
* Precision/recall metrics
* mAP performance

### 4.5 Storing Output

Training results (weights, metrics, plots) are copied back to Google Drive for later evaluation or deployment.

---

## 5. Project Structure Example

```
📂 cricket-ball-tracker (1)
 ┣ 📂 data (1)
 │  ┣ 📂 raw_videos (1)
 │  ┗ 📂 processed_videos (1)
 ┣ 📂 Roboflow
 │  ┣ 📂 train
 │  ┣ 📂 train
 │  ┣ 📂 valid
 │  ┗ data.yaml
 ┣ 📂 runs
 ┣ model_training.ipynb
 ┣ video_preprocessing.ipynb
 ┗ README.md
```

---

## 6. Key Challenges & Solutions

### Challenge: GPU Memory Limit (15 GB)

**Solution:**
Switched to YOLO medium model, reduced batch size, and cleared CUDA cache.

### Challenge: Annotation Consistency

**Solution:**
Validation scripts were used to ensure every image had a corresponding `.txt` label file.

### Challenge: Video Frame Size Mismatch

**Solution:**
Preprocessing pipeline standardized all raw videos.

---

## 7. Final Output

After training, the model is capable of detecting cricket balls across various lighting conditions and backgrounds. The exported model can be used for:

* Live tracking
* Ball trajectory estimation
* Shot classification in cricket analytics
* Automated highlight extraction

---

## 8. Tools & Technologies

* **Roboflow** – Annotation & Dataset Management
* **Google Colab** – GPU-enabled training environment
* **Ultralytics YOLO** – Object detection
* **OpenCV** – Video Frame Processing
* **TensorBoard** – Visual logging


Just tell me.
