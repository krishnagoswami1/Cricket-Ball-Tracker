# Cricket Ball Detection – Prediction Pipeline

This notebook performs inference using a trained YOLO model on cricket videos. It loads the trained weights, processes each frame, applies detections, visualizes annotated frames, and optionally writes a prediction video to disk.

---

## 1. Purpose

The objective of this notebook is to use a trained cricket-ball detection model (YOLO) to:

* Load raw or preprocessed cricket match videos
* Perform frame-by-frame detection
* Draw bounding boxes around detected cricket balls
* Visualize results in real time
* Export an annotated prediction video

---

## 2. Workflow Summary

### 2.1 Importing Dependencies

The notebook loads essential libraries:

* `opencv-python` for video operations
* `numpy` for numerical work
* `ultralytics` YOLO for inference
* Utility libraries for display and output

These tools form the base of the prediction pipeline.

---

### 2.2 Loading the Model

A YOLO model is loaded using:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
```

This loads the final trained checkpoint produced during model training.

---

### 2.3 Reading Video Input

A video is opened using:

```python
cap = cv2.VideoCapture("input_video.mp4")
```

The script extracts:

* Frame width
* Frame height
* FPS

This metadata is used for both display and writing output.

---

### 2.4 Frame-by-Frame Prediction

For each frame:

1. Read the frame from the video capture stream
2. Pass it through the YOLO model
3. Retrieve bounding boxes and confidence scores
4. Draw annotations on the frame
5. Display the result using `cv2.imshow`

This creates a real-time prediction display.

---

### 2.5 Writing Annotated Output Video

If enabled, predictions are written to a video using:

```python
VideoWriter(...)
```

This generates an output file where every frame contains bounding-box annotations of detected cricket balls.

---

### 2.6 Cleanup

Once prediction finishes:

* VideoCapture is released
* VideoWriter is closed
* Windows opened by OpenCV are destroyed

This ensures proper memory and resource management.

---

## 3. Output

The notebook produces:

* Real-time display of detections
* A saved MP4 file with YOLO-annotated frames
* Detection confidence statistics in the console (depending on code)

---

## 4. Key Features

* High-speed frame inference using YOLO
* Smooth display loop using OpenCV
* Correct handling of frame sizes and output formats
* Flexible input/output path management

---

## 5. Requirements

* Python 3.x
* OpenCV
* Ultralytics YOLO
* A GPU-enabled environment for maximum speed (Colab recommended)

---

## 6. Notes (if trying to replicate)

* The same trained weights used in training must be present in the working directory.
* Ensure the FPS and resolution of input/output videos are compatible with your hardware.
* If using Colab, enable GPU acceleration for smooth inference.

---
