from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shared.sklearn_model_service import SklearnModelService
from shared.tcn_model_service import PhaseAwareTCNModelService
from shared.tcn_service import (
    _get_simple_tcn_config,
    load_phase_tcn,
    load_sequence_tcn,
    predict_sequence_tcn,
)


class FakeSklearnModel:
    def predict(self, feature_matrix):
        return np.array([1])

    def predict_proba(self, feature_matrix):
        return np.array([[0.1, 0.9]], dtype=np.float32)


class FakeSklearnModelNoProba:
    def predict(self, feature_matrix):
        return np.array([2])


def test_sklearn_model_service_loads_model_and_predicts_with_confidence(monkeypatch) -> None:
    monkeypatch.setattr("shared.sklearn_model_service.joblib.load", lambda path: FakeSklearnModel())

    service = SklearnModelService("model.pkl")
    predicted_class_id, confidence = service.predict(np.array([1.0, 2.0], dtype=np.float32))

    assert service.loaded is True
    assert predicted_class_id == 1
    assert confidence == pytest.approx(0.9)


def test_sklearn_model_service_handles_missing_model_and_model_without_proba(monkeypatch) -> None:
    monkeypatch.setattr("shared.sklearn_model_service.joblib.load", lambda path: (_ for _ in ()).throw(FileNotFoundError("missing")))
    service = SklearnModelService("missing.pkl")
    assert service.loaded is False
    assert service.predict(np.array([1.0], dtype=np.float32)) == (0, None)

    monkeypatch.setattr("shared.sklearn_model_service.joblib.load", lambda path: FakeSklearnModelNoProba())
    service_no_proba = SklearnModelService("plain.pkl")
    assert service_no_proba.predict(np.array([1.0], dtype=np.float32)) == (2, None)


def test_get_simple_tcn_config_uses_metadata_or_infers_from_state_dict() -> None:
    channels, dropout, use_attention = _get_simple_tcn_config(
        {
            "meta": {"channels": [8, 16], "dropout": 0.2, "use_attention": False},
            "model_state": {},
        }
    )
    assert channels == (8, 16)
    assert dropout == pytest.approx(0.2)
    assert use_attention is False

    inferred_channels, inferred_dropout, inferred_attention = _get_simple_tcn_config(
        {
            "model_state": {
                "tcn.1.conv1.weight": torch.zeros(7, 3, 3),
                "tcn.0.conv1.weight": torch.zeros(5, 3, 3),
                "attention.weight": torch.zeros(1, 7),
            }
        }
    )
    assert inferred_channels == (5, 7)
    assert inferred_dropout == pytest.approx(0.1)
    assert inferred_attention is True


def test_get_simple_tcn_config_falls_back_to_fc_weight_or_defaults() -> None:
    fc_channels, _, fc_attention = _get_simple_tcn_config(
        {"model_state": {"fc.weight": torch.zeros(2, 9)}}
    )
    assert fc_channels == (9,)
    assert fc_attention is False

    default_channels, _, default_attention = _get_simple_tcn_config({"model_state": {}})
    assert default_channels == (64, 64, 64)
    assert default_attention is False


def test_load_sequence_tcn_and_load_phase_tcn_success_and_failure(monkeypatch) -> None:
    created_sequence_models: list[object] = []
    created_phase_models: list[object] = []

    class FakeSequenceModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.loaded_state = None
            self.evaluated = False
            created_sequence_models.append(self)

        def load_state_dict(self, state_dict) -> None:
            self.loaded_state = state_dict

        def eval(self) -> None:
            self.evaluated = True

    class FakePhaseModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.loaded_state = None
            self.evaluated = False
            created_phase_models.append(self)

        def load_state_dict(self, state_dict) -> None:
            self.loaded_state = state_dict

        def eval(self) -> None:
            self.evaluated = True

    checkpoints = {
        "sequence.pt": {
            "in_dim": 4,
            "T": 12,
            "label_map": {"good": 0, "bad": 1},
            "model_state": {"fc.weight": torch.zeros(2, 6)},
            "meta": {"channels": [6], "dropout": 0.15, "use_attention": True},
        },
        "phase.pt": {
            "in_dim": 10,
            "num_classes": 2,
            "window": 7,
            "state_dict": {"fc.weight": torch.zeros(2, 64, 1)},
        },
    }

    def fake_torch_load(path, map_location="cpu"):
        if path == "missing.pt":
            raise RuntimeError("missing")
        return checkpoints[path]

    monkeypatch.setattr("shared.tcn_service.SimpleTCN", FakeSequenceModel)
    monkeypatch.setattr("shared.tcn_service.PhaseTCN", FakePhaseModel)
    monkeypatch.setattr("shared.tcn_service.torch.load", fake_torch_load)

    model, window, inv_labels, in_dim = load_sequence_tcn("sequence.pt")
    assert model is created_sequence_models[0]
    assert window == 12
    assert inv_labels == {0: "good", 1: "bad"}
    assert in_dim == 4
    assert created_sequence_models[0].evaluated is True

    phase_model, phase_window, phase_in_dim = load_phase_tcn("phase.pt")
    assert phase_model is created_phase_models[0]
    assert phase_window == 7
    assert phase_in_dim == 10
    assert created_phase_models[0].evaluated is True

    assert load_sequence_tcn("missing.pt") == (None, None, None, None)
    assert load_phase_tcn("missing.pt") == (None, None, None)


def test_predict_sequence_tcn_returns_label_confidence_and_probabilities(monkeypatch) -> None:
    class FakeSequencePredictor:
        def __call__(self, feature_window_tensor):
            assert feature_window_tensor.shape == (1, 3, 2)
            return torch.tensor([[1.0, 3.0]], dtype=torch.float32)

    monkeypatch.setattr("shared.tcn_service.resample_time", lambda x, target_t: np.ones((target_t, x.shape[1]), dtype=np.float32))

    label, confidence, probabilities = predict_sequence_tcn(
        FakeSequencePredictor(),
        {0: "bad", 1: "good"},
        3,
        np.array([[1.0, 2.0]], dtype=np.float32),
    )

    assert label == "good"
    assert confidence == pytest.approx(float(probabilities[1]))
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)


def test_phase_aware_tcn_model_service_predicts_and_reports_loaded_flags(monkeypatch) -> None:
    sequence_calls: list[str] = []

    def fake_load_sequence(path: str):
        sequence_calls.append(path)
        if path == "bottom.pt":
            return object(), 5, {0: "bad", 1: "good"}, 4
        if path == "stand.pt":
            return object(), 6, {0: "not_ready", 1: "ready"}, 3
        return None, None, None, None

    def fake_predict_sequence(model, inv_labels, target_t, feature_window):
        return inv_labels[1], 0.88, np.array([0.12, 0.88], dtype=np.float32)

    monkeypatch.setattr("shared.tcn_model_service.load_sequence_tcn", fake_load_sequence)
    monkeypatch.setattr("shared.tcn_model_service.predict_sequence_tcn", fake_predict_sequence)
    monkeypatch.setattr("shared.tcn_model_service.os.path.isfile", lambda path: path == "phase.pt")

    class FakePhaseModel:
        def __call__(self, feature_tensor):
            return torch.tensor(
                [[[3.0, 1.0], [1.0, 3.0], [0.5, 4.0]]],
                dtype=torch.float32,
            )

    monkeypatch.setattr(
        "shared.tcn_model_service.load_phase_tcn",
        lambda path: (FakePhaseModel(), 3, 2),
    )

    service = PhaseAwareTCNModelService(
        "bottom.pt",
        stand_path="stand.pt",
        phase_path="phase.pt",
    )

    assert sequence_calls == ["bottom.pt", "stand.pt"]
    assert service.bottom_loaded is True
    assert service.stand_loaded is True
    assert service.phase_loaded is True

    bottom = service.predict_bottom(np.ones((5, 4), dtype=np.float32))
    stand = service.predict_stand(np.ones((6, 3), dtype=np.float32))
    phase_last = service.predict_phase(np.ones((3, 2), dtype=np.float32))
    phase_vote = service.predict_phase(
        np.ones((3, 2), dtype=np.float32),
        decision_mode="majority_vote",
    )

    assert bottom[0] == "good"
    assert stand[0] == "ready"
    assert phase_last == "concentric"
    assert phase_vote == "concentric"


def test_phase_aware_tcn_model_service_returns_unknown_when_models_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "shared.tcn_model_service.load_sequence_tcn",
        lambda path: (None, None, None, None),
    )
    monkeypatch.setattr("shared.tcn_model_service.os.path.isfile", lambda path: False)

    service = PhaseAwareTCNModelService("bottom.pt", phase_path="phase.pt")

    assert service.bottom_loaded is False
    assert service.stand_loaded is False
    assert service.phase_loaded is False
    bottom_label, bottom_confidence, bottom_probabilities = service.predict_bottom(
        np.ones((2, 2), dtype=np.float32)
    )
    stand_label, stand_confidence, stand_probabilities = service.predict_stand(
        np.ones((2, 2), dtype=np.float32)
    )

    assert bottom_label == "unknown"
    assert bottom_confidence == 0.0
    assert bottom_probabilities.size == 0
    assert stand_label == "unknown"
    assert stand_confidence == 0.0
    assert stand_probabilities.size == 0
    assert service.predict_phase(np.ones((2, 2), dtype=np.float32)) == "unknown"
