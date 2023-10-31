"""Provides Machine Learning statistics for model analysis."""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import RocCurveDisplay, classification_report


def print_model_statistics(y_true: np.ndarray, y_preds: np.ndarray, losses: np.ndarray) -> None:
    # Print classification report
    print(classification_report(y_true, y_preds, target_names=['Background', 'Intrusion'],
        digits=3, zero_division=0))
