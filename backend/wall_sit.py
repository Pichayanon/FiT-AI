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
BASE_DIR = "videos/wall_sit"

DATASET = {
    0: [os.path.join(BASE_DIR, "correct", f)
        for f in os.listdir(os.path.join(BASE_DIR, "correct"))
        if f.endswith((".mp4", ".mov"))],

    1: [os.path.join(BASE_DIR, "feet_too_close", f)
        for f in os.listdir(os.path.join(BASE_DIR, "feet_too_close"))
        if f.endswith((".mp4", ".mov"))],
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


def get_best_leg(lm):
    legs = {
        "RIGHT": {
            "hip": mp_pose.PoseLandmark.RIGHT_HIP,
            "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
            "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
            "toe": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        },
        "LEFT": {
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "toe": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        }
    }

    best_leg = None
    best_vis = 0

    for side, idx in legs.items():
        vis = np.mean([lm[i].visibility for i in idx.values()])
        if vis > best_vis:
            best_vis = vis
            best_leg = side

    return best_leg, legs[best_leg] if best_leg else (None, None)


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    foot_wall_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark
        # Decide which side is facing camera
        is_right_view = (
                lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility >
                lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
        )

        if is_right_view:
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        else:
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

        # Foot to wall distance (horizontal)
        foot_wall_dist = abs(ankle.x - hip.x)

        # Normalize by shoulder width (scale invariant)
        shoulder_width = abs(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        )

        foot_wall_norm = foot_wall_dist / (shoulder_width + 1e-6)
        foot_wall_values.append(foot_wall_norm)

    cap.release()

    if len(foot_wall_values) == 0:
        return None

    feat = [
        np.mean(foot_wall_values),
        np.std(foot_wall_values),
        np.min(foot_wall_values),
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
