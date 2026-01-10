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
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

mp_pose = mp.solutions.pose

def extract_features(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV can't open video: {video_path}")

    pose = mp_pose.Pose(static_image_mode=False)

    knee_angles = []
    knee_forward = []

    while True:
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

        knee_ang = angle(hip, knee, ankle)
        forward = knee[0] - toe_x

        knee_angles.append(knee_ang)
        knee_forward.append(forward)

    cap.release()

    if len(knee_angles) == 0:
        raise RuntimeError("No pose landmarks detected. Try better lighting / full body in frame.")

    feat = np.array([
        np.mean(knee_angles),
        np.std(knee_angles),
        np.mean(knee_forward),
        np.std(knee_forward),
        np.max(knee_forward),
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
    predict("videos/IMG_1362.mov")
    # predict("videos/IMG_1359.mp4")
