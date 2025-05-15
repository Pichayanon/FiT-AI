import cv2
import json
import numpy as np
import os
import mediapipe as mp

# Constants
SEQ_LENGTH = 30
FEATURES = 48
IMPORTANT_KEYPOINTS = [
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

mp_pose = mp.solutions.pose

def extract_selected_keypoints(landmarks):
    keypoints = []
    for name in IMPORTANT_KEYPOINTS:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints

def extract_sequence_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    sequence = []

    while cap.isOpened() and len(sequence) < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            keypoints = extract_selected_keypoints(lm)
            sequence.append(keypoints)

    cap.release()
    pose.close()

    # Pad or truncate to SEQ_LENGTH
    if len(sequence) < SEQ_LENGTH:
        if sequence:
            while len(sequence) < SEQ_LENGTH:
                sequence.append(sequence[-1])
        else:
            sequence = [[0.0] * FEATURES for _ in range(SEQ_LENGTH)]
    else:
        sequence = sequence[:SEQ_LENGTH]

    return sequence

def save_sequence_as_json(sequence, output_path):
    with open(output_path, 'w') as f:
        json.dump({"sequence": sequence}, f, indent=2)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_folder = "video"
    output_folder = "samples_data"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            path = os.path.join(input_folder, file)
            sequence = extract_sequence_from_video(path)
            output_file = os.path.splitext(file)[0] + ".json"
            save_sequence_as_json(sequence, os.path.join(output_folder, output_file))
