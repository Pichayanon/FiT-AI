import cv2
import mediapipe as mp
import numpy as np
import joblib

# -----------------------------
# Labels
# -----------------------------
LABELS = {0: "correct", 1: "feet_too_close"}

# -----------------------------
# Utils
# -----------------------------


mp_pose = mp.solutions.pose

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

    feat = np.array([
        np.mean(foot_wall_values),
        np.std(foot_wall_values),
        np.min(foot_wall_values),
    ], dtype=np.float32)

    return feat

# -----------------------------
# Predict
# -----------------------------
def predict(video_path: str, model_path="wall_sit_side_model.pkl"):
    model = joblib.load(model_path)
    feat = extract_features(video_path).reshape(1, -1)

    pred = int(model.predict(feat)[0])
    prob = model.predict_proba(feat)[0]
    conf = float(prob[pred])

    print(f"Video: {video_path}")
    print(f"Prediction: {LABELS.get(pred, pred)}")
    print(f"Confidence: {conf:.3f}")
    print("Features:", feat.flatten())
    print("-" * 40)


if __name__ == "__main__":
    predict("videos/wall_sit/correct/wallsit_correct_1.mp4")
    predict("videos/wall_sit/correct/wallsit_correct_2.mp4")
    predict("videos/wall_sit/feet_too_close/IMG_1359.mov")
