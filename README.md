#  Journalist Detector using YOLOv8

An AI-powered system designed to detect journalists in various scenes by identifying the presence of a person alongside specific equipment like cameras or microphones.

---

##  Project Overview

This project was developed to automatically identify journalists in images or videos using computer vision techniques. By detecting both a person and typical journalist equipment (e.g., camera or microphone), the system can infer the presence of a journalist.

---

##  Technologies Used

- **YOLOv8** – Object detection model
- **Roboflow** – Dataset management and labeling
- **Google Colab** – Model training
- **OpenCV** – Image processing
- **Python** – Programming language

---

##  Dataset

- Combined two datasets:
  - One for detecting people
  - Another for detecting cameras and microphones
- Total images: **~400**
- Labeled and managed via Roboflow
- Split into training (70%), validation (20%), and testing (10%)

---

## ⚙️ How It Works

1. The model is trained to detect three classes: `Person`, `Camera`, and `Microphone`.
2. During inference, if a person is detected along with a camera or microphone in the same image, they are likely to be a journalist.
3. A post-processing script filters only images where both conditions are met.

---

##  Testing & Evaluation

- Trained over **50 epochs**
- Best validation results:
  - `Camera`: mAP50 ~ 17%
  - `Microphone`: mAP50 ~ 2.7%
  - `Person`: Detected successfully but with limited bounding boxes
- Early results show promising potential; further improvement needed with more annotated data

---

##  Future Work

- Increase dataset size with more labeled "Person + Equipment" images
- Improve bounding box accuracy
- Optimize model for real-time detection in video streams
- Create a live demo or web interface

---

##  How to Use

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train23/weights/best.pt")

# Predict
results = model.predict(source="path/to/image_or_video", save=True)
