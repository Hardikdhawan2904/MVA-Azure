"""
app/services/boot_trainer.py — Startup Dataset Change Detector & Auto-Trainer

Called once from app/main.py's lifespan at service startup — previously
called by main.py's AnalyticsAgent.__init__() on every subprocess spawn.
Now that the service is long-lived and every request brings its own
uploaded dataset (see app/routes/analyze.py), this check no longer means
"is *this request's* file newer than the models" — it means "is the local
reference dataset (DATASET_PATH, used by train.py and scripts/cli.py for
local testing) newer than the persisted models", checked once per process
lifetime rather than once per request.

Logic:
    1. Get the modification timestamp of the reference dataset CSV.
    2. Get the timestamp of the oldest trained model file.
    3. If the dataset is NEWER than the oldest model → retrain automatically.
    4. If no models exist → retrain automatically.
    5. If models are newer than the dataset → skip training (already up to date).
"""

import logging
import subprocess
import sys

from app.config import BASE_DIR, DATASET_PATH, ML_MODEL_SAVE_DIR

logger = logging.getLogger("BootTrainer")

# The model artifacts that must exist and be up to date. Includes the
# paired scaler/encoder side-files, not just the 4 primary model files —
# a missing side-file alone used to report "up to date" and the
# persisted-predict paths would then silently fall back to fit-fresh with
# nothing louder than a log line.
_REQUIRED_MODELS = [
    "lightgbm_underwriting_result_actual.pkl",
    "isolation_forest.pkl",
    "isolation_forest_scaler.pkl",
    "xgboost_classifier.pkl",
    "xgboost_label_encoder.pkl",
    "kmeans.pkl",
    "kmeans_scaler.pkl",
]

# Ceiling on how long a startup retrain is allowed to block service
# startup — a hung train.py (e.g. a stuck Prophet/cmdstan step) shouldn't
# be able to hang startup indefinitely.
_TRAINING_TIMEOUT_SECONDS = 300


def _get_dataset_mtime() -> float:
    """Return the modification time (epoch seconds) of the reference dataset CSV."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    return DATASET_PATH.stat().st_mtime


def _get_oldest_model_mtime() -> float | None:
    """
    Return the modification time of the OLDEST required model file.
    If any model file is missing, returns None (forces retraining).
    """
    mtimes = []
    for model_name in _REQUIRED_MODELS:
        model_path = ML_MODEL_SAVE_DIR / model_name
        if not model_path.exists():
            logger.warning(f"Required model missing: {model_name}")
            return None
        mtimes.append(model_path.stat().st_mtime)
    return min(mtimes)  # Return oldest model timestamp


def _run_training() -> bool:
    """
    Execute train.py as a subprocess and stream its log output.
    Returns True if training succeeded, False otherwise.
    """
    train_script = BASE_DIR / "train.py"
    logger.info("=" * 60)
    logger.info("Startup Auto-Trainer: New dataset detected!")
    logger.info("Starting model retraining — this takes ~15 seconds...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=False,  # Stream output directly to console
            text=True,
            timeout=_TRAINING_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Auto-Training TIMED OUT after {_TRAINING_TIMEOUT_SECONDS}s.")
        logger.error("The service will continue using the previous model versions.")
        return False

    if result.returncode == 0:
        logger.info("=" * 60)
        logger.info("Auto-Training complete. Models are now up to date.")
        logger.info("=" * 60)
        return True
    else:
        logger.error(f"Auto-Training FAILED with exit code {result.returncode}.")
        logger.error("The service will continue using the previous model versions.")
        return False


def check_and_retrain_if_needed() -> bool:
    """
    Main entry point, called once from app/main.py's lifespan at startup.

    Returns:
        True  — if retraining was triggered and succeeded (or skipped as up to date).
        False — if retraining was triggered but failed.
    """
    try:
        dataset_mtime = _get_dataset_mtime()
        oldest_model_mtime = _get_oldest_model_mtime()

        if oldest_model_mtime is None:
            # At least one model file is missing — must train
            logger.info("Startup check: One or more model files missing. Training required.")
            return _run_training()

        if dataset_mtime > oldest_model_mtime:
            # Dataset is newer than the models — retrain
            from datetime import datetime
            dataset_ts = datetime.fromtimestamp(dataset_mtime).strftime("%Y-%m-%d %H:%M:%S")
            model_ts   = datetime.fromtimestamp(oldest_model_mtime).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"Startup check: Dataset updated ({dataset_ts}) is newer than "
                f"oldest model ({model_ts}). Auto-retraining..."
            )
            return _run_training()
        else:
            # Models are up to date — skip retraining
            logger.info("Startup check: Models are up to date. No retraining needed. ✓")
            return True

    except FileNotFoundError as e:
        logger.error(f"Startup check skipped: {e}")
        return False
    except Exception as e:
        logger.error(f"Startup check error: {e}. Skipping auto-retrain.")
        return False
