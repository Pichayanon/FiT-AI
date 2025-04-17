import cv2
import mediapipe as mp
import numpy as np
import os
import csv

SEQ_LENGTH = 30
FEATURES = 48
VIDEO_FOLDER = "video"
OUTPUT_CSV = "squat_data.csv"

mp_pose = mp.solutions.pose

IMPORTANT_KEYPOINTS = [
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def extract_selected_keypoints(landmarks):
    keypoints = []
    for name in IMPORTANT_KEYPOINTS:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints


def extract_squats(video_path, label, writer):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    capturing = False
    stage = None
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            keypoints = extract_selected_keypoints(lm)

            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(hip, knee, ankle)

            if angle < 90 and not capturing:
                capturing = True
                sequence = []
                stage = "Down"

            if capturing:
                sequence.append(keypoints)

            if angle > 160 and stage == "Down" and capturing:
                stage = "Up"
                capturing = False

                if len(sequence) >= 10:
                    while len(sequence) < SEQ_LENGTH:
                        sequence.append(sequence[-1])
                    flattened = np.array(sequence[:SEQ_LENGTH]).flatten().tolist()
                    flattened.append(label)
                    writer.writerow(flattened)

    cap.release()


def batch_extract():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        for filename in os.listdir(VIDEO_FOLDER):
            if filename.endswith(".mp4"):
                label = 0 if "incorrect" in filename.lower() else 1
                path = os.path.join(VIDEO_FOLDER, filename)
                print(f"Processing: {filename} | Label: {label}")
                extract_squats(path, label, writer)
        print("All done! Data saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    batch_extract()
