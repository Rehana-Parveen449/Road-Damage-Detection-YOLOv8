# 🚀 Road Defect Detection using YOLOv8 Segmentation

## 📌 Overview

This project presents an end-to-end computer vision solution for detecting and segmenting different types of road defects using YOLOv8 Nano Segmentation models.

Instead of training a single model for all road conditions, three specialized segmentation models were developed independently for:

- Road Crack Detection
- Pothole Detection
- Road Sign & Road Marking Detection

The trained models were integrated into a Streamlit web application that allows users to upload road images, select a detection model, and visualize segmentation results in real time.

---

## 🎯 Objectives

- Detect road surface cracks using image segmentation.
- Identify potholes for road maintenance applications.
- Detect road signs and road markings.
- Build an interactive Streamlit application for inference.
- Demonstrate an end-to-end deep learning workflow from training to deployment.

---

## ✨ Features

- YOLOv8 Nano Segmentation models
- Three independent detection models
- Interactive Streamlit interface
- Image upload functionality
- Adjustable confidence threshold
- Detection summary
- Individual detection details
- Segmentation visualization
- Modular architecture for multiple detection tasks

---

## 🧠 Models Used

| Model | Purpose |
|--------|----------|
| YOLOv8n-seg | Road Crack Detection |
| YOLOv8n-seg | Pothole Detection |
| YOLOv8n-seg | Road Sign & Road Marking Detection |

---

## 📂 Dataset

The project was developed using three independently curated datasets for road defect detection.

The datasets include annotated images for:

- Road surface cracks
- Potholes
- Road signs
- Road markings

The datasets were customized and organized using Roboflow to maintain a consistent annotation format compatible with YOLOv8 segmentation training.

---

## 🏗️ Project Structure

```text
Road-Damage-Detection-YOLOv8
│
├── app
│   ├── app.py
│   └── requirements.txt
│
├── notebooks
│   ├── crack_detection_training.ipynb
│   ├── pothole_detection_training.ipynb
│   └── road_sign_detection_training.ipynb
│
├── models
│
├── results
│   ├── crack_detection
│   ├── pothole_detection
│   └── road_sign_detection
│
├── screenshots
│
├── .gitignore
└── README.md
```

---

## 📊 Model Performance

| Model | Precision | Recall | mAP50 | mAP50-95 |
|--------|----------:|-------:|------:|----------:|
| Crack Detection | 0.760 | 0.692 | 0.705 | 0.433 |
| Pothole Detection | 0.882 | 0.653 | 0.715 | 0.490 |
| Road Sign & Road Marking Detection | 0.877 | 0.799 | 0.845 | 0.644 |

---

## 💻 Streamlit Application

The application enables users to:

- Upload road images
- Select a detection model
- Adjust confidence threshold
- Perform segmentation
- Visualize predicted results
- Review detected object details

---

## 🛠️ Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- PyTorch
- Google Colab
- Roboflow

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Rehana-Parveen449/Road-Damage-Detection-YOLOv8.git
```

Install dependencies:

```bash
pip install -r app/requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## 📈 Future Improvements

- Merge all detections into a single multi-class model
- Improve model accuracy using larger datasets
- Add video inference support
- Deploy the application to Streamlit Cloud
- Integrate GPS-based road damage reporting

---

## 👩‍💻 Author

**Rehana Praveen**

Aspiring Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/Rehana-Parveen449
