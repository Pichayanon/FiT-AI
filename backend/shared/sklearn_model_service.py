"""
Sklearn model service for loading and serving predictions.

Used by exercises with joblib-serialized sklearn classifiers
(plank, wall_sit).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import joblib
import numpy as np


class SklearnModelService:
    """Load and serve predictions from a joblib-serialized sklearn classifier.

    Supports models with both predict() and predict_proba() methods.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the model service and attempt to load the model.

        Args:
            model_path: Path to the joblib model file.
        """
        self.model_path = model_path
        self.model = self._load()

    def _load(self) -> Any:
        """Load the model from disk.

        Returns:
            Loaded model object, or None if loading fails.
        """
        try:
            model = joblib.load(self.model_path)
            print(f"[MODEL] Loaded: {self.model_path}")
            return model
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[MODEL] Cannot load model: {exc}")
            return None

    @property
    def loaded(self) -> bool:
        """Return True if the model is loaded successfully."""
        return self.model is not None

    def predict(self, feat: np.ndarray) -> Tuple[int, Optional[float]]:
        """Run prediction on a feature vector.

        Args:
            feat: Feature array. Will be reshaped to (1, -1) if needed.

        Returns:
            Tuple of (predicted_class_id, confidence_or_None).
            Confidence is the predicted class probability if the model
            supports predict_proba(), otherwise None.
        """
        if self.model is None:
            return 0, None

        feat_2d = feat.reshape(1, -1)
        pred = int(self.model.predict(feat_2d)[0])
        conf: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feat_2d)[0]
            conf = float(proba[pred])
        return pred, conf
