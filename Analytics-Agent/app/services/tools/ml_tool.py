"""
tools/ml_tool.py — Tool 7: ML Tool Orchestrator

Routes ML requests to the appropriate sub-module based on the query type.
Enforces ML readiness check before any model execution.

Routes:
  forecast      → ml/forecaster.py (Prophet or LightGBM)
  anomaly       → ml/anomaly_detector.py (Isolation Forest)
  classify      → ml/classifier.py (XGBoost)
  segment       → ml/classifier.py (K-Means)
"""

import logging

import pandas as pd

from app.config import ML_READINESS_THRESHOLD
from app.services.ml.forecaster       import ProphetForecaster, LightGBMForecaster
from app.services.ml.anomaly_detector import AnomalyDetector
from app.services.ml.classifier       import VarianceClassifier, RiskSegmenter

logger = logging.getLogger(__name__)


class MLTool:
    """
    Orchestrates all ML operations.
    Checks ML readiness score before executing any model.
    """

    def __init__(self, ml_readiness_score: float = 99.75):
        self.ml_readiness_score = ml_readiness_score
        self._prophet    = ProphetForecaster()
        self._lgbm       = LightGBMForecaster()
        self._anomaly    = AnomalyDetector()
        self._classifier = VarianceClassifier()
        self._segmenter  = RiskSegmenter()
        logger.info(f"ML Tool initialised: readiness_score={ml_readiness_score}")

    # ── Readiness Gate ────────────────────────────────────────────────────────

    def _check_readiness(self) -> bool:
        """
        Enforce ML readiness gate.
        If score < threshold, block ML execution and recommend fallback.
        """
        if self.ml_readiness_score < ML_READINESS_THRESHOLD:
            logger.warning(
                f"ML readiness score {self.ml_readiness_score} < threshold "
                f"{ML_READINESS_THRESHOLD}. ML execution blocked."
            )
            return False
        return True

    def _readiness_error(self) -> dict:
        return {
            "error": (
                f"ML readiness score ({self.ml_readiness_score:.1f}) is below the "
                f"required threshold ({ML_READINESS_THRESHOLD}). "
                f"Falling back to deterministic Analytics Tool. "
                f"Ensure data quality is improved before re-running ML."
            ),
            "fallback": "Use Analytics Tool for trend/variance analysis",
        }

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast_timeseries(
        self,
        ts_df: pd.DataFrame,
        periods: int = 6,
        freq: str = "ME",
    ) -> dict:
        """
        Time-series forecast using Prophet.

        Args:
            ts_df   : DataFrame with ['ds', 'y'] columns.
            periods : Months to forecast ahead.
            freq    : Pandas frequency string.

        Returns: Prophet forecast dict.
        """
        if not self._check_readiness():
            return self._readiness_error()

        logger.info(f"MLTool → ProphetForecaster: {len(ts_df)} historical points, "
                    f"{periods} periods ahead")
        return self._prophet.fit_and_forecast(ts_df, periods=periods, freq=freq)

    def forecast_with_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        categorical_cols: list[str] | None = None,
    ) -> dict:
        """
        Multi-feature regression forecast using LightGBM.

        Args:
            df             : Full DataFrame.
            target_col     : Column to predict.
            feature_cols   : Input feature columns.
            categorical_cols: Categorical columns for native encoding.

        Returns: LightGBM regression result with feature importances.
        """
        if not self._check_readiness():
            return self._readiness_error()

        logger.info(f"MLTool → LightGBMForecaster: target='{target_col}', "
                    f"{len(feature_cols)} features")
        return self._lgbm.fit_and_predict(df, target_col, feature_cols, categorical_cols)

    # ── Anomaly Detection ─────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
        contamination: float = 0.05,
    ) -> dict:
        """
        Detect ratio anomalies using Isolation Forest.

        Args:
            df           : DataFrame with ratio columns.
            feature_cols : Ratio columns to analyse.
            label_cols   : ID columns to include in output.
            contamination: Expected anomaly fraction.

        Returns: Anomaly detection results.
        """
        if not self._check_readiness():
            return self._readiness_error()

        persisted = self._anomaly.score(df, feature_cols, label_cols)
        if persisted is not None:
            logger.info(f"MLTool → IsolationForest (persisted): {len(df)} records")
            return persisted

        logger.info(f"MLTool → IsolationForest (fit-fresh, no persisted model found): "
                    f"{len(df)} records, contamination={contamination}")
        return self._anomaly.detect(df, feature_cols, contamination, label_cols)

    # ── Classification ────────────────────────────────────────────────────────

    def classify_variance_drivers(
        self,
        df: pd.DataFrame,
        target_col: str = "primary_variance_driver",
    ) -> dict:
        """
        Train XGBoost to classify variance drivers.

        Args:
            df         : DataFrame with variance + label columns.
            target_col : Target classification column.

        Returns: Classification results with feature importances.
        """
        if not self._check_readiness():
            return self._readiness_error()

        logger.info(f"MLTool → XGBoostClassifier: target='{target_col}', "
                    f"{len(df)} records")
        return self._classifier.fit_and_evaluate(df, target_col)

    def predict_variance_driver(self, record: dict) -> dict:
        """
        Predict variance driver for a single record with SHAP explanation.

        Args:
            record: Dict of feature values.

        Returns: Predicted driver + SHAP feature attributions.
        """
        if not self._check_readiness():
            return self._readiness_error()
        return self._classifier.predict_with_explanation(record)

    def get_root_cause_ml_evidence(self, df: pd.DataFrame) -> dict | None:
        """
        ML-predicted variance driver + SHAP evidence for a root-cause query,
        using the persisted XGBoost classifier — corroborating evidence
        alongside the deterministic RootCauseTool breakdown, not a
        replacement for it. Returns None if ML isn't authorized or no
        persisted model exists yet (caller should skip this evidence).
        """
        if not self._check_readiness():
            return None
        return self._classifier.predict_batch_with_explanation(df)

    def get_forecast_key_drivers(self, target_col: str) -> dict | None:
        """
        Persisted LightGBM feature importances for target_col — "what
        typically drives this KPI" evidence alongside Prophet's time-series
        trend. Returns None if ML isn't authorized or no persisted model
        exists yet for this target (caller should skip this evidence).
        """
        if not self._check_readiness():
            return None
        return self._lgbm.get_key_drivers(target_col)

    # ── Segmentation ──────────────────────────────────────────────────────────

    def segment_portfolio(
        self,
        df: pd.DataFrame,
        n_clusters: int = 3,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
    ) -> dict:
        """
        Cluster portfolio into risk segments using K-Means.

        Args:
            df          : DataFrame with ratio columns.
            n_clusters  : Number of clusters (default 3).
            feature_cols: KPI ratio columns to cluster on.
            label_cols  : ID columns to include in output.

        Returns: Cluster assignments + centroid profiles.
        """
        if not self._check_readiness():
            return self._readiness_error()

        persisted = self._segmenter.assign(df, feature_cols, label_cols)
        if persisted is not None:
            logger.info(f"MLTool → KMeans (persisted): {len(df)} records")
            return persisted

        logger.info(f"MLTool → KMeans (fit-fresh, no persisted model found): "
                    f"{len(df)} records, {n_clusters} clusters")
        return self._segmenter.segment(df, feature_cols, n_clusters, label_cols)

    # ── Model Registry ────────────────────────────────────────────────────────

    def get_readiness_status(self) -> dict:
        """Return current ML readiness status."""
        return {
            "ml_readiness_score":  self.ml_readiness_score,
            "threshold":           ML_READINESS_THRESHOLD,
            "ml_authorized":       self._check_readiness(),
            "available_models": [
                "Prophet (time-series forecast)",
                "LightGBM (multi-feature regression)",
                "IsolationForest (anomaly detection)",
                "XGBoost (variance driver classification)",
                "K-Means (risk segmentation)",
            ],
        }
