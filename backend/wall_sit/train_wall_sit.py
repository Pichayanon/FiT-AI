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
BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "dataset",
    "wall_sit"
)
BASE_DIR = os.path.abspath(BASE_DIR)

DATASET = {
    0: "correct",
    1: "feet_too_close",
    2: "feet_too_far",
    3: "back_of_wall",
    4: "not_deep_enough",
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

    foot_wall_vals = []
    knee_angles = []
    torso_alignments = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark

        # Decide visible side
        right_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
        left_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
        is_right = right_vis > left_vis

        if is_right:
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        else:
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # 1️⃣ Foot to wall distance (horizontal)
        foot_wall_dist = abs(ankle.x - hip.x)

        shoulder_width = abs(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        )

        foot_wall_norm = foot_wall_dist / (shoulder_width + 1e-6)
        foot_wall_vals.append(foot_wall_norm)

        # 2️⃣ Knee angle
        knee_angle = angle(
            [hip.x, hip.y],
            [knee.x, knee.y],
            [ankle.x, ankle.y]
        )
        knee_angles.append(knee_angle)

        # 3️⃣ Torso alignment (back against wall)
        torso_align = abs(shoulder.x - hip.x)
        torso_alignments.append(torso_align)

    cap.release()

    if len(foot_wall_vals) == 0:
        return None

    feat = [
        np.mean(foot_wall_vals),
        np.std(foot_wall_vals),
        np.mean(knee_angles),
        np.min(knee_angles),
        np.mean(torso_alignments),
    ]

    return feat


# -----------------------------
# Build Dataset
# -----------------------------
X, y = [], []

for label, folder in DATASET.items():
    folder_path = os.path.join(BASE_DIR, folder)
    videos = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith((".mp4", ".mov"))
    ]

    for v in videos:
        feat = extract_features(v)
        if feat is not None:
            X.append(feat)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

# -----------------------------
# Train model
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000
    ))
])

model.fit(X, y)

joblib.dump(model, "wall_sit_model.pkl")
print("Model saved: wall_sit_model.pkl")
