"""
ml/persistence.py — shared pickle loader for the ML tools.

Query-time code (AnomalyDetector, RiskSegmenter, VarianceClassifier,
LightGBMForecaster) predicts against the models train.py already fit on the
full dataset, instead of re-fitting from scratch on whatever rows a single
query happens to filter down to. This is the one place that knows how to
load those persisted artifacts.
"""

import logging
import pickle
from typing import Any

from app.config import ML_MODEL_SAVE_DIR

logger = logging.getLogger(__name__)


def load_pickle(filename: str) -> Any | None:
    """Load a pickle from ML_MODEL_SAVE_DIR. Returns None if the file is
    missing or fails to unpickle — callers use this to fall back to
    fitting fresh rather than erroring on a cold start."""
    path = ML_MODEL_SAVE_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load persisted model {path}: {e}")
        return None
