# Shoplifting Detection Using Deep Learning

## Overview
This project focuses on detecting shoplifting behavior using a deep learning model trained on video datasets. The model analyzes video frames and determines whether a shoplifting incident has occurred.
![UI Screenshot](https://github.com/MalakAmgad/shoplifters/blob/main/detection/static/images/Screenshot%20(910).png)
![UI Screenshot](https://github.com/MalakAmgad/shoplifters/blob/main/detection/static/images/Screenshot%20(911).png)

## Dataset
- **Shoplifter Videos**: Contains clips of individuals engaging in shoplifting behavior.
- **Non-Shoplifter Videos**: Normal shopping behavior without theft.
- Data was collected and preprocessed to ensure uniform frame sizes.

## Preprocessing
- Extract 25 frames per video.
- Resize frames to **112x112 pixels**.
- Normalize pixel values.
- Convert video frames into a numpy array suitable for TensorFlow.
- Motion detection using background subtraction.
- Optical flow analysis for frame stabilization.
- Selecting high-motion frames for improved accuracy.

### Preprocessing Code Sample
```python
import cv2
import numpy as np
import os
from tqdm import tqdm

# Constants
TARGET_WIDTH = 112
TARGET_HEIGHT = 112
TARGET_FRAME_COUNT = 25

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def preprocess_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame = frame.astype(np.float32) / 255.0  # Normalize
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    np.save(os.path.join(output_folder, os.path.basename(video_path).replace('.mp4', '.npy')), frames)
```

## Model Architecture
The deep learning model is a **3D Convolutional Neural Network (CNN)** designed to process video frames.

### Layers:
- **Conv3D + MaxPooling3D** (Feature extraction from video frames)
- **Flatten** (Converts extracted features into a dense format)
- **Dense + Dropout** (Final classification layers)

### Model Code Sample
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', input_shape=(25, 112, 112, 3)),
    MaxPooling3D((2, 2, 2)),
    Conv3D(64, (3, 3, 3), activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Model achieves **99% accuracy** on validation data.

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

## User Interface (UI)
- The frontend is designed with a **modern UI**, featuring a **background image with a shadow overlay**.
- Users can upload video files for shoplifting detection.
- Upon detection, the UI displays the **predicted label** and provides sample **frames extracted** from the video.
- The interface ensures a seamless experience for video analysis.
- **Detection Output**: When shoplifting is detected, the system highlights the **detected shoplifter frames** in the UI.
- **Address Logging**: The UI includes an input field to store the **shop location** where the video was recorded.

## Running the Project
### Install Dependencies
```sh
pip install django tensorflow opencv-python numpy
```
### Start Server
```sh
python manage.py runserver
```

## Project Structure
```
shoplifter_detection/
│── detection/                  # Django app for model inference
│   ├── static/                  # Static files (CSS, images)
│   ├── templates/
│   ├── views.py                   # Model inference logic
│   ├── urls.py                    # API routes
│   ├── model_1.h5                 # Your trained model
│── shoplifter_detection/          # Main Django project
│── media/                        # Stores uploaded videos
│── manage.py                      # Django management script
│── db.sqlite3                      # Database
│── venv/                          # Virtual environment
```

## Results
- **Confusion Matrix**: Shows model performance.
- **ROC Curve**: High area under curve indicates strong classification ability.
- **Sample Frames Displayed**: UI presents extracted frames when a shoplifting incident is detected.

## Visuals
![CNF](https://github.com/MalakAmgad/shoplifters/blob/main/detection/static/images/cnf3.png)
![processing](https://github.com/MalakAmgad/shoplifters/blob/main/detection/static/images/original.png)

## Future Improvements
- Expand dataset for improved generalization.
- Optimize model for real-time inference.
- Improve UI responsiveness for better user experience.


