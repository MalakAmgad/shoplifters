import os
import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Load TensorFlow model
MODEL_PATH = "detection/model_1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Model's expected input dimensions
EXPECTED_FRAMES = 25  # Match model input
FRAME_HEIGHT = 112     # Resize to match model
FRAME_WIDTH = 112      # Resize to match model

def index(request):
    """Render the homepage."""
    return render(request, "index.html")

import tensorflow as tf

def detect(request):
    print("\n[INFO] Received request for shoplifting detection.")
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        frames, error = extract_frames(video_file)
        print(f"[DEBUG] Video file received: {video_file.name}")

        if error:
            print(f"[ERROR] Preprocessing failed: {error}")
            return render(request, "index.html", {"detected": "Preprocessing Failed"})

        frames = np.expand_dims(frames, axis=0)
        logits = model.predict(frames)[0]  # Raw output

        # Convert logits to probabilities (if needed)
        probabilities = tf.nn.softmax(logits).numpy()
        print(f"Softmax Probabilities: {probabilities}")

        shoplifting_prob = probabilities[1]  # Ensure this is correct
        threshold = 0.6

        detected = "Shoplifter Detected" if shoplifting_prob >= threshold else "No Shoplifting Detected"
        return render(request, "index.html", {"detected": detected})

    return render(request, "index.html", {"detected": None})

@csrf_exempt
def process_video(request):
    """Handle video upload, process frames, and return prediction as JSON."""
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        frames, error = extract_frames(video_file)

        if error:
            return JsonResponse({"error": error}, status=400)

        frames = np.expand_dims(frames, axis=0)  # Shape: (1, 25, 112, 112, 3)
        prediction = model.predict(frames).tolist()

        return JsonResponse({"prediction": prediction})

    return JsonResponse({"error": "No video uploaded"}, status=400)
import os
from django.conf import settings

def extract_frames(video_file):
    """Extracts 25 frames from the video file."""
    try:
        print("\n[INFO] Extracting frames from video...")

        # Ensure temp directory exists
        temp_dir = os.path.join(settings.BASE_DIR, "media/temp_videos")
        os.makedirs(temp_dir, exist_ok=True)

        # Define full path for saving video
        file_path = os.path.join(temp_dir, video_file.name)

        # Save file
        with open(file_path, "wb") as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        print(f"[DEBUG] Temporary file saved at: {file_path}")

        # Open video for processing
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
           print(f"[ERROR] OpenCV failed to open: {file_path}")
           return None, "Failed to open video file"
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert to RGB (Django uploads are often BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to 112x112
            frame = cv2.resize(frame, (112, 112))

            # Normalize pixels
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

            if frame_count % 5 == 0:
                print(f"[DEBUG] Processed Frame {frame_count}: {frame.shape}")

        cap.release()
        os.remove(file_path)  # Clean up temp file
        print(f"[INFO] Total frames extracted: {len(frames)}")

        # Ensure 25 frames
        if len(frames) >= 25:
            return np.array(frames[:25]), None
        elif len(frames) > 0:
            while len(frames) < 25:
                frames.append(frames[-1])  # Pad with last frame
            print("[DEBUG] Frames padded to 25.")
            return np.array(frames), None

        print("[ERROR] No frames extracted from video.")
        return None, "No frames extracted"

    except Exception as e:
        print(f"[ERROR] Video processing failed: {str(e)}")
        return None, f"Error processing video: {str(e)}"
