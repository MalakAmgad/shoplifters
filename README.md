# Shoplifting Detection Using Deep Learning

## Overview
This project focuses on detecting shoplifting behavior using a deep learning model trained on video datasets. The model analyzes video frames and determines whether a shoplifting incident has occurred.

## Dataset
- **Shoplifter Videos**: Contains clips of individuals engaging in shoplifting behavior.
- **Non-Shoplifter Videos**: Normal shopping behavior without theft.
- Data was collected and preprocessed to ensure uniform frame sizes.

## Model Architecture
The deep learning model is a **3D Convolutional Neural Network (CNN)** designed to process video frames.

### Layers:
- **Conv3D + MaxPooling3D** (Feature extraction from video frames)
- **Flatten** (Converts extracted features into a dense format)
- **Dense + Dropout** (Final classification layers)

Model achieves **99% accuracy** on validation data.

## Preprocessing
- Extract 25 motion frames per video.
- Resize frames to **112x112 pixels**.
- Normalize pixel values.
- Convert video frames into a numpy array suitable for TensorFlow.

## Deployment (Django)
### Steps:
1. **Setup Django Project**
   ```sh
   django-admin startproject shoplifter_detection
   cd shoplifter_detection
   python manage.py startapp detection
   ```
2. **Load Model**: The TensorFlow model (`model_1.h5`) is integrated into Django views.
3. **Extract Frames**: OpenCV is used to process uploaded video files.
4. **Prediction API**: The backend processes video input and returns a prediction.

## Running the Project
### Install Dependencies
```sh
pip install django tensorflow opencv-python numpy
```
### Start Server
```sh
python manage.py runserver
```

## Results
- **Confusion Matrix**: Shows model performance.
- **ROC Curve**: High area under curve indicates strong classification ability.

## Visuals
(Include model architecture diagram, confusion matrix, and sample predictions here)

## Future Improvements
- Expand dataset for improved generalization.
- Optimize model for real-time inference.

