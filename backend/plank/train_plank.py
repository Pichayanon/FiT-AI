import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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


# -----------------------------
# Utils
# -----------------------------
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


mp_pose = mp.solutions.pose


def extract_features(video_path: str):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()

    hip_signed_distances = []
    body_angles = []
    normalized_hip_heights = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark

        # เลือกด้านที่ visibility สูงกว่า
        right_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
        left_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
        is_right = right_vis > left_vis

        if is_right:
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        else:
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

        shoulder_xy = np.array([shoulder.x, shoulder.y])
        hip_xy = np.array([hip.x, hip.y])
        ankle_xy = np.array([ankle.x, ankle.y])

        # -------------------------
        # 1️⃣ Signed distance ของ hip จากเส้น shoulder-ankle
        # -------------------------
        line_vec = ankle_xy - shoulder_xy
        hip_vec = hip_xy - shoulder_xy

        signed_dist = np.cross(line_vec, hip_vec) / (np.linalg.norm(line_vec) + 1e-6)
        hip_signed_distances.append(signed_dist)

        # -------------------------
        # 2️⃣ Body angle (shoulder-hip-ankle)
        # -------------------------
        body_angle = angle(shoulder_xy, hip_xy, ankle_xy)
        body_angles.append(body_angle)

        # -------------------------
        # 3️⃣ Normalized hip height
        # -------------------------
        body_length = np.linalg.norm(ankle_xy - shoulder_xy) + 1e-6
        hip_height_norm = (hip.y - 0.5*(shoulder.y + ankle.y)) / body_length
        normalized_hip_heights.append(hip_height_norm)

    cap.release()

    if len(hip_signed_distances) == 0:
        return None

    feat = [
        np.mean(hip_signed_distances),
        np.std(hip_signed_distances),

        np.mean(normalized_hip_heights),
        np.std(normalized_hip_heights),

        np.mean(body_angles),
        np.std(body_angles),
    ]

    return feat



# -----------------------------
# Build Dataset
# -----------------------------
X, y = [], []

for label, folder in DATASET.items():
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    videos = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".mov"))
    ]

    for v in videos:
        feat = extract_features(v)
        if feat is not None:
            X.append(feat)
            y.append(label)

if not X:
    raise RuntimeError(f"No valid videos found under {BASE_DIR}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Plank dataset shape:", X.shape)
print("Classes:", np.unique(y))


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

model.fit(X, y)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")

