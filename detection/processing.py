import cv2
import numpy as np
import os

# Constants
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
TARGET_FRAME_COUNT = 30

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def compute_optical_flow(prev_frame, curr_frame):
    """Extracts motion features using Optical Flow."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def preprocess_video_django(video_input):
    """Preprocess video from either Django file object or file path."""
    try:
        # Check if input is a Django file object or a file path
        if hasattr(video_input, "chunks"):  # Django file object
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb+") as f:
                for chunk in video_input.chunks():
                    f.write(chunk)
            video_path = temp_path
        else:  # Already a file path
            video_path = video_input

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Unable to read video."

        extracted_frames = []
        motion_frames = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and normalize frame
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32) / 255.0  # Normalize

            if prev_frame is not None:
                # Compute optical flow between consecutive frames
                flow = compute_optical_flow(prev_frame, frame)
                motion_score = np.linalg.norm(flow) / (TARGET_WIDTH * TARGET_HEIGHT)

                if motion_score > 1.5:  # Adaptive motion threshold
                    motion_frames.append(frame)

            prev_frame = frame
            extracted_frames.append(frame)

        cap.release()

        # Delete temp file only if it was created
        if video_input == temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # Use motion frames if available, otherwise fallback to extracted frames
        selected_frames = motion_frames if motion_frames else extracted_frames

        # Ensure exactly 30 frames
        frame_count = len(selected_frames)
        if frame_count >= TARGET_FRAME_COUNT:
            final_frames = np.array(selected_frames[:TARGET_FRAME_COUNT])
        else:
            last_frame = selected_frames[-1] if selected_frames else np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.float32)
            final_frames = np.array(selected_frames)
            while len(final_frames) < TARGET_FRAME_COUNT:
                final_frames = np.concatenate([final_frames, last_frame[np.newaxis, ...]], axis=0)

        final_frames = np.expand_dims(final_frames, axis=0)  # Add batch dimension (1, 30, 224, 224, 3)

        return final_frames, None

    except Exception as e:
        return None, f"Error processing video: {e}"
