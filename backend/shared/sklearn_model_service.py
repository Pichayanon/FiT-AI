from __future__ import annotations

from typing import Any, Optional, Tuple

import joblib
import numpy as np


class SklearnModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = self._load()

    def _load(self) -> Any:
        try:
            model = joblib.load(self.model_path)
            print(f"[MODEL] Loaded: {self.model_path}")
            return model
        except Exception as exc:
            print(f"[MODEL] Cannot load model: {exc}")
            return None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray) -> Tuple[int, Optional[float]]:
        if self.model is None:
            return 0, None

        feature_matrix = features.reshape(1, -1)
        predicted_class_id = int(self.model.predict(feature_matrix)[0])
        confidence: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_matrix)[0]
            confidence = float(probabilities[predicted_class_id])
        return predicted_class_id, confidence
