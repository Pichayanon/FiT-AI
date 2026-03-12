"""
Label mapper for mapping class indices to human-readable strings.

Used by exercises with discrete class predictions to convert
numeric prediction IDs into display labels.
"""

from __future__ import annotations

from typing import Dict


class LabelMapper:
    """Map class indices to human-readable label strings."""

    def __init__(self, labels: Dict[int, str]) -> None:
        """Initialize with a mapping of class index to label string.

        Args:
            labels: Dictionary mapping integer class IDs to label strings.
        """
        self.labels = labels

    def label_of(self, pred_id: int) -> str:
        """Return the label string for the given predicted class ID.

        Args:
            pred_id: Predicted class index.

        Returns:
            Label string, or the string representation of pred_id if not found.
        """
        return self.labels.get(int(pred_id), str(pred_id))
