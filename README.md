# 🎥 Real-Time Object Detection using OpenCV & MobileNet-SSD

![Python](https://img.shields.io/badge/Python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 💡 Project Overview

This project implements a **real-time object detection system** using **OpenCV’s DNN module** and the **MobileNet-SSD (Single Shot Detector)** deep learning model.

The system captures live video from a webcam and detects common real-world objects such as **persons, cars, dogs, bottles, chairs, and more**, displaying bounding boxes with confidence scores in real time.

The project is lightweight, fast, and ideal for understanding **deep learning inference in computer vision applications**.

---

## 🧠 Detection Pipeline

### 🔹 Live Video Capture
- Uses OpenCV’s `VideoCapture` for real-time webcam streaming
- Horizontal frame flipping for a mirror-view experience

### 🔹 Deep Learning Model
- MobileNet-SSD pretrained on the PASCAL VOC dataset
- Fast and efficient object detection
- Confidence-based filtering to remove weak detections

### 🔹 Frame Processing
- Frames resized to `300 × 300`
- Mean subtraction and scaling using `cv2.dnn.blobFromImage`
- Optimized for real-time performance

### 🔹 Visualization
- Bounding boxes around detected objects
- Class labels with confidence percentages
- Clean, readable real-time output

---

## 🛠️ Tech Stack

- **Python 3.13**
- **OpenCV (cv2)**
- **NumPy**
- **MobileNet-SSD (Caffe Model)**

---

## 📦 Install Dependencies

Run the following command in your terminal:

```bash
pip install opencv-python numpy
```

## ▶️ Run the Application

Execute the Python script:

```bash
python object_detection.py
```

🔴 Press q to close the detection window.

## 📁 Project Structure
```text 
Live-Object-Detection/
│
├── object_detection.py
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
└── README.md
```

### 🎯 Use Cases

- Real-time surveillance systems
- Human and object monitoring
- Computer vision learning projects
- AI-powered camera applications
- Academic and demonstration projects

### 🔮 Future Enhancements

This project can be extended to include:
🎥 Video file and IP camera support
⚡ GPU acceleration (CUDA / OpenCL)
📊 Object counting and analytics
🌐 Web-based interface using Flask or Streamlit
🧠 Custom object training

✨ Conclusion

This project demonstrates hands-on skills in:
- Real-time computer vision
- Deep learning model deployment
- OpenCV DNN framework
- Clean and structured Python programming

If you find this project useful or inspiring, feel free to ⭐ star the repository!
