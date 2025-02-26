import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Frame rate (25 FPS) and target duration (2 seconds, which is 50 frames)
FPS = 25
TARGET_FRAMES = 50  # 2 seconds * 25 frames per second

def trim_or_pad_video(video_path):
    """
    Trim or pad the video to ensure it has TARGET_FRAMES frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    frames = []
    while len(frames) < TARGET_FRAMES and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"No frames were read from {video_path}.")
        return []

    # If the video is shorter than the target length, pad by duplicating the last frame
    while len(frames) < TARGET_FRAMES:
        frames.append(frames[-1].copy())  # Duplicate the last valid frame

    # If the video is longer than the target length, trim it
    frames = frames[:TARGET_FRAMES]

    return frames

def extract_landmarks(video_path):
    """
    Extract landmarks from a video.
    """
    frames = trim_or_pad_video(video_path)
    if not frames:
        return []

    landmark_data = []
    for frame in frames:
        # Convert the frame to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        frame_landmarks = []

        # Extract pose landmarks (33 keypoints)
        if results.pose_landmarks:
            pose_keypoints = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.pose_landmarks.landmark
            ]
        else:
            pose_keypoints = [[0, 0, 0]] * 33
        frame_landmarks.extend(pose_keypoints)

        # Extract left hand landmarks (21 keypoints)
        if results.left_hand_landmarks:
            left_hand_keypoints = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.left_hand_landmarks.landmark
            ]
        else:
            left_hand_keypoints = [[0, 0, 0]] * 21
        frame_landmarks.extend(left_hand_keypoints)

        # Extract right hand landmarks (21 keypoints)
        if results.right_hand_landmarks:
            right_hand_keypoints = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.right_hand_landmarks.landmark
            ]
        else:
            right_hand_keypoints = [[0, 0, 0]] * 21
        frame_landmarks.extend(right_hand_keypoints)

        # Convert to numpy array for consistency
        frame_landmarks = np.array(frame_landmarks)

        # Append landmarks for this frame
        landmark_data.append(frame_landmarks.tolist())

    return landmark_data

# Preprocess New Data (Normalize Landmarks)
def preprocess_new_data(X):
    """
    Preprocess the skeleton data for inference.
    - Flatten and normalize the keypoints for each frame.
    """
    processed_X = []
    seq_array = np.array(X)  # Convert sequence to numpy array
    if seq_array.ndim == 3:  # Ensure it's a 3D array (frames, landmarks, dimensions)
        # Flatten keypoints per frame (e.g., 75 x 3 -> 225)
        flattened_seq = seq_array.reshape(seq_array.shape[0], -1)
        # Normalize the landmarks (per frame), handle zero vectors
        normalized_seq = np.nan_to_num(flattened_seq / np.linalg.norm(flattened_seq, axis=-1, keepdims=True))
        processed_X.append(normalized_seq)
    else:
        raise ValueError(f"Invalid shape for input sequence: {seq_array.shape}")
    return np.array(processed_X)

# Sliding Window Function
def apply_sliding_window(X, window_size=50, step_size=25):
    """
    Apply sliding window to new data.
    """
    X_windows = []
    num_frames = X.shape[0]
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X[start:end])  # Extract window
    return np.array(X_windows)

# Predict Function (with confidence scores, displaying only if confidence > 50%)
def predict_skeleton_sequence_with_confidence(model, label_encoder, X, window_size=50, step_size=25):
    """
    Predict the class of a skeleton sequence using the pre-trained model.
    Additionally, provide confidence scores for each predicted class.
    Only display predictions with confidence greater than 50%.
    """
    # Preprocess the sequence
    X_processed = preprocess_new_data(X)[0]  # Preprocess and get the first (only) sequence
    # Apply sliding window
    X_windows = apply_sliding_window(X_processed, window_size, step_size)
    # Add batch dimension for prediction
    X_windows = np.expand_dims(X_windows, axis=-1)  # Shape: (num_windows, window_size, flattened_keypoints)

    # Predict using the model
    predictions = model.predict(X_windows)  # Shape: (num_windows, num_classes)

    # Get predicted classes and their confidence scores
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)  # Maximum probability for each prediction

    # Decode the predicted classes
    decoded_predictions = label_encoder.inverse_transform(predicted_classes)
    
    # Combine predictions and confidence scores
    results = []
    for decoded_prediction, confidence in zip(decoded_predictions, confidence_scores):
        if confidence > 0.50:  # Only keep predictions with confidence > 50%
            results.append({
                "predicted_class": decoded_prediction,
                "confidence_score": confidence
            })

    return results

# Main workflow for prediction
if _name_ == "_main_":
    # Path to the saved model, label encoder
    model_path = "/Users/visheshbishnoi/Desktop/ISL/ISL/stcgn_isl_model_with_lstm_tuned.keras"
    label_encoder_path = "/Users/visheshbishnoi/Desktop/ISL/ISL/label_encoder.json"
    
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load the label encoder
    with open(label_encoder_path, "r") as file:
        label_classes = json.load(file)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_classes)

    # Path to the video file
    video_path =  "/Users/visheshbishnoi/Desktop/ISL/ISL/exxx/MVI_3055.MOV"
    
    # Extract skeleton data from video
    skeleton_data = extract_landmarks(video_path)
    if not skeleton_data:
        raise ValueError("No landmarks extracted from the video.")

    # Predict using the skeleton data
    predictions_with_confidence = predict_skeleton_sequence_with_confidence(
        model, label_encoder, skeleton_data
    )
    
    # Display predictions and confidence scores
    if predictions_with_confidence:
        for i, result in enumerate(predictions_with_confidence):
            print(f"Window {i + 1}: Class = {result['predicted_class']}, Confidence = {result['confidence_score']:.2f}")
    else:
        print("No predictions with confidence above 50% were found.")
