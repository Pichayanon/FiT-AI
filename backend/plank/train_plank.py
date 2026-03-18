import os

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plank.features import aggregate_window, choose_visible_side, extract_frame_features


# -----------------------------
# Config
# -----------------------------
PLANK_DIR = os.path.dirname(__file__)

BASE_DIR = os.path.join(
    PLANK_DIR,
    "..",
    "dataset",
    "plank",
)
BASE_DIR = os.path.abspath(BASE_DIR)

MODEL_DIR = os.path.join(PLANK_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plank_model.pkl")

# Map numeric labels -> folder names under BASE_DIR
# Adjust folders to match your actual dataset structure.
DATASET = {
    0: "correct",
    1: "hips_too_high",
    2: "hips_too_low",
}


mp_pose = mp.solutions.pose


def extract_video_features(video_path: str):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()

    frame_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(image_rgb)
        if not pose_result.pose_landmarks:
            continue

        landmarks = pose_result.pose_landmarks.landmark
        side = choose_visible_side(landmarks)
        frame_features.append(extract_frame_features(landmarks, side))

    cap.release()
    pose.close()

    if not frame_features:
        return None

    return aggregate_window(frame_features).tolist()



# -----------------------------
# Build Dataset
# -----------------------------
feature_matrix, label_array = [], []

for label, folder in DATASET.items():
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    videos = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".mov"))
    ]

    for video_path in videos:
        feature_vector = extract_video_features(video_path)
        if feature_vector is not None:
            feature_matrix.append(feature_vector)
            label_array.append(label)

if not feature_matrix:
    raise RuntimeError(f"No valid videos found under {BASE_DIR}")

feature_matrix = np.array(feature_matrix, dtype=np.float32)
label_array = np.array(label_array, dtype=np.int64)

print("Plank dataset shape:", feature_matrix.shape)
print("Classes:", np.unique(label_array))


# -----------------------------
# Train model
# -----------------------------
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=2000,
            ),
        ),
    ]
)

model.fit(feature_matrix, label_array)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")
