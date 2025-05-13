import cv2
import mediapipe as mp
import numpy as np
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SEQ_LENGTH = 30
FEATURES = 48

model = load_model("squat_tcn_model.h5")

mp_drawing = mp.solutions.drawing_utils
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
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def extract_selected_keypoints(landmarks):
    keypoints = []
    for name in IMPORTANT_KEYPOINTS:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints


def is_fully_visible(results):
    if not results.pose_landmarks:
        return False
    landmarks = results.pose_landmarks.landmark
    for name in ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        if lm.visibility < 0.5 or lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
            return False
    return True


def draw_ui(frame, counter, stage, feedback_msg=None, knee_angle=None):
    cv2.rectangle(frame, (0, 0), (550, 120), (245, 117, 16), -1)
    cv2.putText(frame, "REPS", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, str(counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(frame, "STAGE", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, stage if stage else "", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if feedback_msg:
        cv2.putText(frame, feedback_msg, (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if knee_angle is not None:
        cv2.putText(frame, f"Knee angle: {int(knee_angle)} deg", (250, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    counter = 0
    stage = None
    capturing = False
    sequence = []
    feedback = ""
    knee_angle = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Camera not found.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if is_fully_visible(results):
                try:
                    landmarks = results.pose_landmarks.landmark

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    knee_angle = calculate_angle(hip, knee, ankle)

                    keypoints = extract_selected_keypoints(landmarks)

                    if knee_angle < 90 and not capturing:
                        capturing = True
                        sequence = []
                        stage = "Down"
                        feedback = ""

                    if capturing:
                        sequence.append(keypoints)

                    if knee_angle > 160 and stage == "Down" and capturing:
                        stage = "Up"
                        capturing = False

                        if len(sequence) >= 10:
                            while len(sequence) < SEQ_LENGTH:
                                sequence.append(sequence[-1])
                            input_data = np.array(sequence[:SEQ_LENGTH]).reshape(1, SEQ_LENGTH, FEATURES)
                            pred = model.predict(input_data, verbose=0)[0][0]

                            if pred > 0.8:
                                counter += 1
                                feedback = "Good squat!"
                            else:
                                feedback = "Bad form!"

                except Exception as e:
                    print("Error:", e)

            draw_ui(image, counter, stage, feedback, knee_angle)
            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3))

            cv2.imshow("Squat Checker", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
