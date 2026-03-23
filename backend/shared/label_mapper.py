from __future__ import annotations

from typing import Dict


class LabelMapper:
    def __init__(self, labels: Dict[int, str]) -> None:
        self.labels = labels

    def label_of(self, pred_id: int) -> str:
        return self.labels.get(int(pred_id), str(pred_id))
