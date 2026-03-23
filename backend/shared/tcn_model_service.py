"""Shared TCN model service helpers for streaming sessions."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from shared.tcn_service import (
    load_phase_tcn,
    load_sequence_tcn,
    predict_sequence_tcn,
)


_PHASE_LABELS: Dict[int, str] = {0: "eccentric", 1: "concentric"}


class PhaseAwareTCNModelService:
    """Load and serve bottom/stand/phase TCN models for streaming inference."""

    def __init__(
        self,
        bottom_path: str,
        *,
        stand_path: Optional[str] = None,
        phase_path: Optional[str] = None,
    ) -> None:
        self.bottom_model, self.bottom_T, self.inv_labels_bottom, self.bottom_in_dim = (
            load_sequence_tcn(bottom_path)
        )

        self.stand_model = None
        self.stand_T = None
        self.inv_labels_stand = None
        self.stand_in_dim = None
        if stand_path:
            (
                self.stand_model,
                self.stand_T,
                self.inv_labels_stand,
                self.stand_in_dim,
            ) = load_sequence_tcn(stand_path)

        self.phase_model = None
        self.phase_window = None
        self.phase_in_dim = None
        if phase_path and os.path.isfile(phase_path):
            self.phase_model, self.phase_window, self.phase_in_dim = load_phase_tcn(phase_path)

    @property
    def bottom_loaded(self) -> bool:
        return self.bottom_model is not None

    @property
    def stand_loaded(self) -> bool:
        return self.stand_model is not None

    @property
    def phase_loaded(self) -> bool:
        return self.phase_model is not None and self.phase_window is not None

    def predict_bottom(self, window_features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run bottom TCN prediction. Returns (label, confidence, probs)."""
        if self.bottom_model is None or self.bottom_T is None:
            return "unknown", 0.0, np.array([])
        return predict_sequence_tcn(
            self.bottom_model,
            self.inv_labels_bottom,
            int(self.bottom_T),
            window_features.astype(np.float32),
        )

    def predict_stand(self, window_features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run standing TCN prediction when a stand model is available."""
        if self.stand_model is None or self.stand_T is None or self.inv_labels_stand is None:
            return "unknown", 0.0, np.array([])
        return predict_sequence_tcn(
            self.stand_model,
            self.inv_labels_stand,
            int(self.stand_T),
            window_features.astype(np.float32),
        )

    def predict_phase(
        self,
        window_features: np.ndarray,
        decision_mode: str = "last_logits",
    ) -> str:
        """Run phase TCN prediction using the configured decision mode."""
        if (
            self.phase_model is None
            or self.phase_window is None
            or window_features.shape[0] < self.phase_window
        ):
            return "unknown"

        phase_window_size = int(self.phase_window)
        phase_feature_window = window_features[-phase_window_size:].astype(np.float32)
        phase_feature_tensor = torch.from_numpy(phase_feature_window).unsqueeze(0)

        with torch.no_grad():
            logits = self.phase_model(phase_feature_tensor)[0]
            if decision_mode == "majority_vote":
                timestep_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                vote_counts = np.bincount(
                    timestep_predictions,
                    minlength=len(_PHASE_LABELS),
                )
                predicted_phase_index = int(np.argmax(vote_counts))
            else:
                predicted_phase_index = int(torch.argmax(logits[-1, :]).item())

        return _PHASE_LABELS.get(predicted_phase_index, "unknown")
