"""
ml/anomaly_detector.py — Isolation Forest Anomaly Detection

Model 3: Isolation Forest (scikit-learn)
  - Unsupervised — no labelled anomaly data required
  - Detects financial ratio outliers (loss ratio, combined ratio, etc.)
  - Returns anomaly score and flag per record

Why Isolation Forest?
- No labelled anomalies exist in the dataset — so supervised models can't be used.
- Isolation Forest is the industry standard for unsupervised tabular anomaly detection.
- Computationally lightweight on 2,000 rows.
- Anomaly score provides a ranking of severity, not just a binary flag.
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from app.config import MODEL_REGISTRY_PATH, ANOMALY_CONTAMINATION
from app.services.ml.persistence import load_pickle

logger = logging.getLogger(__name__)

# Ratio columns to monitor for anomalies
RATIO_FEATURE_COLUMNS = [
    "loss_ratio_actual",
    "expense_ratio_actual",
    "combined_ratio_actual",
    "claim_frequency_actual",
    "average_claim_severity_actual",
    "reserve_adequacy_ratio_actual",
    "policy_lapse_rate_actual",
    "renewal_rate_actual",
    "premium_collection_rate_actual",
]


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for insurance financial ratios.
    """

    def __init__(self):
        self.model = None
        self.feature_cols = []

    def score(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
    ) -> dict | None:
        """
        Score records against the persisted (train.py-fit) IsolationForest —
        no fitting here. Returns None if no persisted model exists yet, so
        the caller can fall back to detect() (fit-fresh) on a cold start.

        Scoring against one model fit on the full dataset, rather than a new
        model fit on whatever rows this query's filters returned, is what
        makes anomaly scores comparable across queries instead of being a
        different noisy fit every time.
        """
        model = load_pickle("isolation_forest.pkl")
        scaler = load_pickle("isolation_forest_scaler.pkl")
        if model is None or scaler is None:
            return None

        # The scaler must see exactly the columns it was fit on, in the same
        # order — unlike detect() (fit-fresh), we can't silently narrow to
        # whatever subset happens to be in df. If this query's data is
        # missing a column the persisted model expects, fall back to
        # fit-fresh (which adapts to whatever's available) rather than
        # erroring on a feature-name mismatch.
        cols = list(getattr(scaler, "feature_names_in_", feature_cols or RATIO_FEATURE_COLUMNS))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.info(f"Persisted IsolationForest expects columns missing from this query's "
                        f"data ({missing}) — falling back to fit-fresh.")
            return None

        df_feat = df[cols].copy().apply(pd.to_numeric, errors="coerce")
        df_feat = df_feat.dropna(how="all")
        valid_idx = df_feat.index
        df_feat = df_feat.fillna(df_feat.median())

        if df_feat.empty:
            return {"error": "No scoreable rows after filtering."}

        X = scaler.transform(df_feat)
        labels = model.predict(X)
        scores = model.score_samples(X)

        result_df = df.loc[valid_idx].copy()
        result_df["anomaly_flag"] = labels
        result_df["anomaly_score"] = np.round(scores, 4)

        anomalies = result_df[result_df["anomaly_flag"] == -1].sort_values("anomaly_score")
        output_cols = [c for c in (list(label_cols or []) + cols + ["anomaly_score"]) if c in anomalies.columns]

        return {
            "model":            "IsolationForest",
            "source":           "persisted",
            "total_records":    len(result_df),
            "anomaly_count":    len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / len(result_df) * 100, 2),
            "feature_columns":  cols,
            "anomalies":        anomalies[output_cols].round(4).to_dict(orient="records"),
        }

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        contamination: float = ANOMALY_CONTAMINATION,
        label_cols: list[str] | None = None,
    ) -> dict:
        """
        Detect anomalies in ratio columns using Isolation Forest.

        Args:
            df            : DataFrame with ratio columns.
            feature_cols  : Columns to use as features (defaults to RATIO_FEATURE_COLUMNS).
            contamination : Expected fraction of anomalies (default 5%).
            label_cols    : Columns to include in output for identification
                            (e.g. ['reporting_date', 'region']).

        Returns:
            {
              "model": "IsolationForest",
              "anomaly_count": int,
              "total_records": int,
              "anomaly_rate_pct": float,
              "anomalies": [{"anomaly_score": ..., <label_cols>..., <ratio_cols>...}],
              "feature_columns": [...],
            }
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

        cols = feature_cols or [c for c in RATIO_FEATURE_COLUMNS if c in df.columns]
        if not cols:
            return {"error": "No ratio columns found in DataFrame for anomaly detection."}

        df_feat = df[cols].copy()
        df_feat = df_feat.apply(pd.to_numeric, errors="coerce")

        # Drop rows where ALL feature columns are null
        df_feat = df_feat.dropna(how="all")
        valid_idx = df_feat.index

        # Impute column medians for partial nulls
        df_feat = df_feat.fillna(df_feat.median())

        if len(df_feat) < 10:
            return {
                "error": f"Insufficient rows for anomaly detection: {len(df_feat)} (need >= 10)"
            }

        logger.info(f"IsolationForest: fitting on {len(df_feat)} rows, "
                    f"{len(cols)} features, contamination={contamination}")

        scaler = StandardScaler()
        X = scaler.fit_transform(df_feat)

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        model.fit(X)
        self.model = model
        self.feature_cols = cols

        labels = model.predict(X)          # 1 = normal, -1 = anomaly
        scores = model.score_samples(X)    # More negative = more anomalous

        result_df = df.loc[valid_idx].copy()
        result_df["anomaly_flag"]  = labels
        result_df["anomaly_score"] = np.round(scores, 4)

        anomalies = result_df[result_df["anomaly_flag"] == -1].copy()
        anomalies = anomalies.sort_values("anomaly_score")  # Most anomalous first

        # Build output records
        output_cols = list(label_cols or []) + cols + ["anomaly_score"]
        output_cols = [c for c in output_cols if c in anomalies.columns]

        anomaly_records = anomalies[output_cols].round(4).to_dict(orient="records")

        result = {
            "model":            "IsolationForest",
            "contamination":    contamination,
            "total_records":    len(result_df),
            "anomaly_count":    len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / len(result_df) * 100, 2),
            "feature_columns":  cols,
            "anomalies":        anomaly_records,
        }

        self._update_registry("isolation_forest", {
            "algorithm":        "IsolationForest",
            "anomaly_count":    len(anomalies),
            "total_records":    len(result_df),
            "contamination":    contamination,
            "timestamp":        datetime.utcnow().isoformat(),
        })

        logger.info(f"IsolationForest: {len(anomalies)} anomalies detected "
                    f"({result['anomaly_rate_pct']}%)")
        return result

    @staticmethod
    def _update_registry(key: str, metadata: dict):
        try:
            with open(MODEL_REGISTRY_PATH, "r") as f:
                registry = json.load(f)
            registry["models"][key] = metadata
            with open(MODEL_REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Registry update failed: {e}")
