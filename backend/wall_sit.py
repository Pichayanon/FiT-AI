import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Config
# -----------------------------
DATASET = {
    0: ["videos/IMG_1357.mov", "videos/IMG_1358.mov", "videos/IMG_1361.mov", "videos/IMG_1367.mov"],
    1: ["videos/IMG_1359.mov", "videos/IMG_1360.mov", "videos/IMG_1362.mov", "videos/IMG_1368.mov"],
}


# -----------------------------
# Utils
# -----------------------------
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


mp_pose = mp.solutions.pose


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()

    knee_angles = []
    knee_forward = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark

        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
               lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                 lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        toe_x = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x

        knee_angle = angle(hip, knee, ankle)
        forward = knee[0] - toe_x  # + = เข่าล้ำหน้าเท้า

        knee_angles.append(knee_angle)
        knee_forward.append(forward)

    cap.release()

    if len(knee_angles) == 0:
        return None

    # Statistical features ต่อคลิป
    feat = [
        np.mean(knee_angles),
        np.std(knee_angles),
        np.mean(knee_forward),
        np.std(knee_forward),
        np.max(knee_forward),
    ]
    return feat


# -----------------------------
# Build Dataset
# -----------------------------
X, y = [], []

for label, videos in DATASET.items():
    for v in videos:
        feat = extract_features(v)
        if feat is not None:
            X.append(feat)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", y)

# -----------------------------
# Train model
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

model.fit(X, y)

joblib.dump(model, "wall_sit_side_model.pkl")
print("Model saved: wall_sit_side_model.pkl")
