import cv2
import mediapipe as mp
import numpy as np
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

SEQ_LENGTH = 30
FEATURES = 48
NUM_CLASSES = 2

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
    h, w = frame.shape[:2]
    padding = int(w * 0.015)
    text_y = int(h * 0.05)
    spacing = int(w * 0.08)

    cv2.rectangle(frame, (0, 0), (int(w * 0.43), int(h * 0.17)), (245, 117, 16), -1)

    cv2.putText(frame, "REPS", (padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, str(counter), (padding, text_y + int(h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    cv2.putText(frame, "STAGE", (padding + spacing, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, stage if stage else "", (padding + spacing, text_y + int(h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if feedback_msg:
        cv2.putText(frame, feedback_msg, (padding + spacing * 2, text_y + int(h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if knee_angle is not None:
        cv2.putText(frame, f"Knee angle: {int(knee_angle)} deg", (padding + spacing * 2, text_y + int(h * 0.13)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def main():
    video_path = "video/squat_incorrect_2.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

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
                break

            frame_height, frame_width = frame.shape[:2]

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if is_fully_visible(results):
                try:
                    landmarks = results.pose_landmarks.landmark

                    # LEFT side
                    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle_left = calculate_angle(hip_left, knee_left, ankle_left)

                    # RIGHT side
                    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    angle_right = calculate_angle(hip_right, knee_right, ankle_right)

                    # Take the smaller angle (deeper squat)
                    knee_angle = min(angle_left, angle_right)

                    keypoints = extract_selected_keypoints(landmarks)

                    if knee_angle < 140 and not capturing:
                        capturing = True
                        sequence = []
                        stage = "Down"
                        feedback = ""

                    if capturing:
                        sequence.append(keypoints)

                    if knee_angle > 150 and stage == "Down" and capturing:
                        stage = "Up"
                        capturing = False

                        if len(sequence) >= 10:
                            while len(sequence) < SEQ_LENGTH:
                                sequence.append(sequence[-1])
                            input_data = np.array(sequence[:SEQ_LENGTH]).reshape(1, SEQ_LENGTH, FEATURES)
                            pred = model.predict(input_data, verbose=0)[0]
                            predicted_class = np.argmax(pred)
                            print(pred, "-> class:", predicted_class)

                            if predicted_class == 1:
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

            scale_factor = 1280 / frame_width
            display_frame = cv2.resize(image, (int(frame_width * scale_factor), int(frame_height * scale_factor)))

            cv2.imshow("Squat Checker (Video)", display_frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()