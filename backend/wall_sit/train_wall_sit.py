import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wall_sit.features import aggregate_window_features, extract_frame_features


@dataclass(frozen=True)
class WallSitConfig:
    base_dir: str
    label_map: Dict[int, str]
    test_size: float = 0.3
    random_state: int = 921

    @property
    def classes(self) -> List[int]:
        return sorted(self.label_map.keys())


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "wall_sit")
)

DATASET = WallSitConfig(
    base_dir=BASE_DIR,
    label_map={
        0: "correct",
        1: "feet_too_close",
        2: "feet_too_far",
        3: "back_of_wall",
        4: "not_deep_enough",
    },
)

mp_pose = mp.solutions.pose


class WallSitFeatureExtractor:
    def __init__(self) -> None:
        self._pose = mp_pose.Pose()

    def extract(self, video_path: str) -> Optional[List[float]]:
        capture = cv2.VideoCapture(video_path)

        frame_features: List[Tuple[float, float, float, float]] = []

        while capture.isOpened():
            frame_ok, frame = capture.read()
            if not frame_ok:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self._pose.process(image_rgb)
            if not pose_result.pose_landmarks:
                continue

            landmarks = pose_result.pose_landmarks.landmark

            right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility
            left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility
            side = "right" if right_visibility > left_visibility else "left"
            frame_features.append(extract_frame_features(landmarks, side))

        capture.release()

        if not frame_features:
            return None

        return aggregate_window_features(frame_features).tolist()


class WallSitTrainer:
    def __init__(self, config: WallSitConfig, max_iter: int) -> None:
        self.config = config
        self.max_iter = max_iter
        self.extractor = WallSitFeatureExtractor()
        self.model: Optional[Pipeline] = None

    def build_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        feature_matrix: List[List[float]] = []
        label_array: List[int] = []

        print(f"[DATA] Loading videos from {self.config.base_dir}...")
        for label, folder in self.config.label_map.items():
            folder_path = os.path.join(self.config.base_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"  [WARN] Folder not found: {folder_path}")
                continue

            videos = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith((".mp4", ".MOV", ".mov"))
            ]

            for video_path in videos:
                feature_vector = self.extractor.extract(video_path)
                if feature_vector is not None:
                    feature_matrix.append(feature_vector)
                    label_array.append(label)

        feature_matrix_array = np.asarray(feature_matrix, dtype=np.float32)
        label_array_values = np.asarray(label_array, dtype=np.int64)

        print("Dataset shape:", feature_matrix_array.shape)
        print("Classes:", np.unique(label_array_values))
        return feature_matrix_array, label_array_values

    def create_model(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        multi_class="multinomial",
                        solver="lbfgs",
                        max_iter=self.max_iter,
                    ),
                ),
            ]
        )

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)}")

        model = self.create_model()
        print(
            f"[TRAIN] Fitting model on {len(X_train)} samples "
            f"with max_iter={self.max_iter}..."
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"[TRAIN] Accuracy: {train_acc:.4f}")

        print("\n[VAL] Evaluating on validation set...")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        unique_labels = sorted(np.unique(y))
        target_names = [self.config.label_map[i] for i in unique_labels]

        print("\n[VAL] Confusion matrix (rows=true, cols=pred):")
        print(cm)

        print("\n[VAL] Classification Report:")
        print(
            classification_report(
                y_val,
                y_pred,
                target_names=target_names,
                digits=3,
            )
        )

        print("\n[VAL] Per-class Accuracy:")
        cm_diag = cm.diagonal()
        cm_sum = cm.sum(axis=1)
        for i, label_idx in enumerate(unique_labels):
            class_name = self.config.label_map[label_idx]
            cls_acc = cm_diag[i] / cm_sum[i] if cm_sum[i] > 0 else 0.0
            print(f"  {class_name:<20}: {cls_acc:.4f} " f"({cm_diag[i]}/{cm_sum[i]})")

        print(f"\n[VAL] Overall Accuracy: {acc:.4f}")

        self.model = model
        return model

    def save_model(self, model: Pipeline) -> str:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "wall_sit_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
        return model_path

    def run(self) -> None:
        X, y = self.build_dataset()
        model = self.train_and_evaluate(X, y)
        self.save_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wall Sit Classification Model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max iterations for Logistic Regression (default: 2000)",
    )
    args = parser.parse_args()

    trainer = WallSitTrainer(config=DATASET, max_iter=args.epochs)
    trainer.run()


if __name__ == "__main__":
    main()
