"""app/services/analyzers/anomaly_strategies.py — Stage 7 "Improve Current
Fallbacks" (plan "zany-giggling-crayon"): anomaly-detection strategies
registered in config/model_registry.yml alongside Isolation Forest.

Every strategy shares one interface — `.detect(df, feature_cols,
label_cols=None)` — and returns the same shape AnomalyDetector.detect()
already produces (model/total_records/anomaly_count/anomaly_rate_pct/
feature_columns/anomalies), one row per flagged record with a combined
multi-feature anomaly_score — not the old per-column-per-row nested-loop
shape handle_anomaly()'s deterministic fallback used. That old shape is
specific to Insurance's two hardcoded ratio columns; this generic version
is what Phase 4 wires the new pipeline to, with the Insurance plugin's
`get_preferred_deterministic_strategy("anomaly_detection") == "Z-Score"`
re-verified for exact-match behavior at that point (Phase 3 only builds
and tests the analyzers themselves, not the live wiring — see the plan's
Phase 3/4 split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    cols = [c for c in feature_cols if c in df.columns]
    feat = df[cols].apply(pd.to_numeric, errors="coerce")
    feat = feat.dropna(how="all")
    valid_idx = feat.index
    feat = feat.fillna(feat.median())
    return feat, valid_idx


def _build_result(
    model_name: str,
    df: pd.DataFrame,
    valid_idx: pd.Index,
    flag: np.ndarray,
    score: np.ndarray,
    cols: list[str],
    label_cols: list[str] | None,
    extra: dict | None = None,
) -> dict:
    result_df = df.loc[valid_idx].copy()
    result_df["anomaly_flag"] = flag
    result_df["anomaly_score"] = np.round(score, 4)

    anomalies = result_df[result_df["anomaly_flag"] == -1].sort_values("anomaly_score")
    output_cols = [c for c in (list(label_cols or []) + cols + ["anomaly_score"]) if c in anomalies.columns]

    return {
        "evidence": {
            "model": model_name,
            "total_records": len(result_df),
            "anomaly_count": len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / len(result_df) * 100, 2) if len(result_df) else 0.0,
            "feature_columns": cols,
            "anomalies": anomalies[output_cols].round(4).to_dict(orient="records"),
            **(extra or {}),
        }
    }


class ZScoreStrategy:
    """Row flagged when any feature's |z-score| exceeds the threshold —
    the deterministic default (min_rows: 3, always available)."""

    def detect(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, threshold: float = 3.0) -> dict:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return {"error": "No feature columns found in DataFrame for Z-Score anomaly detection."}
        feat, valid_idx = _feature_frame(df, cols)
        if feat.empty:
            return {"error": "No scoreable rows after filtering."}

        std = feat.std(ddof=0).replace(0, np.nan)
        z = (feat - feat.mean()) / std
        z = z.fillna(0.0)
        max_abs_z = z.abs().max(axis=1).to_numpy()
        flag = np.where(max_abs_z > threshold, -1, 1)

        return _build_result("Z-Score", df, valid_idx, flag, -max_abs_z, cols, label_cols, {"threshold": threshold})


class IQRStrategy:
    """Row flagged when any feature falls outside Q1 - k*IQR .. Q3 + k*IQR
    — robust to non-normal distributions where Z-Score over-flags."""

    def detect(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, multiplier: float = 1.5) -> dict:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return {"error": "No feature columns found in DataFrame for IQR anomaly detection."}
        feat, valid_idx = _feature_frame(df, cols)
        if feat.empty:
            return {"error": "No scoreable rows after filtering."}

        q1, q3 = feat.quantile(0.25), feat.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        outside = ((feat < lower) | (feat > upper)).fillna(False)
        row_flagged = outside.any(axis=1).to_numpy()
        flag = np.where(row_flagged, -1, 1)
        # Distance beyond the nearest violated bound, normalised by IQR —
        # larger = further outside the fence = more anomalous.
        dist = pd.concat([(lower - feat) / iqr, (feat - upper) / iqr], axis=1).max(axis=1).fillna(0).to_numpy()

        return _build_result("IQR", df, valid_idx, flag, dist, cols, label_cols, {"iqr_multiplier": multiplier})


class ModifiedZScoreStrategy:
    """Median Absolute Deviation-based z-score — robust to outliers
    skewing the mean/std the plain Z-Score strategy relies on."""

    def detect(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, threshold: float = 3.5) -> dict:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return {"error": "No feature columns found in DataFrame for Modified Z-Score anomaly detection."}
        feat, valid_idx = _feature_frame(df, cols)
        if feat.empty:
            return {"error": "No scoreable rows after filtering."}

        median = feat.median()
        mad = (feat - median).abs().median().replace(0, np.nan)
        modified_z = (0.6745 * (feat - median) / mad).fillna(0.0)
        max_abs = modified_z.abs().max(axis=1).to_numpy()
        flag = np.where(max_abs > threshold, -1, 1)

        return _build_result("Modified Z-Score (MAD)", df, valid_idx, flag, -max_abs, cols, label_cols, {"threshold": threshold})


class LocalOutlierFactorStrategy:
    """scikit-learn LocalOutlierFactor — density-based, catches local
    outliers Isolation Forest's global partitioning can miss."""

    def detect(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, contamination: float = 0.05) -> dict:
        from sklearn.neighbors import LocalOutlierFactor

        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return {"error": "No feature columns found in DataFrame for LOF anomaly detection."}
        feat, valid_idx = _feature_frame(df, cols)
        if len(feat) < 10:
            return {"error": f"Insufficient rows for LOF anomaly detection: {len(feat)} (need >= 10)"}

        n_neighbors = min(20, len(feat) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        flag = lof.fit_predict(feat)
        score = lof.negative_outlier_factor_

        return _build_result("Local Outlier Factor", df, valid_idx, flag, -score, cols, label_cols, {"contamination": contamination})


class OneClassSVMStrategy:
    """scikit-learn OneClassSVM — expensive (RBF kernel, O(n^2)-ish fit),
    registered cost_tier: expensive; the Scheduler's expensive-operation
    budget (Stage 5) is what keeps this from running unbounded."""

    def detect(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, nu: float = 0.05) -> dict:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return {"error": "No feature columns found in DataFrame for One-Class SVM anomaly detection."}
        feat, valid_idx = _feature_frame(df, cols)
        if len(feat) < 10:
            return {"error": f"Insufficient rows for One-Class SVM anomaly detection: {len(feat)} (need >= 10)"}

        X = StandardScaler().fit_transform(feat)
        model = OneClassSVM(nu=nu, kernel="rbf")
        model.fit(X)
        flag = model.predict(X)
        score = model.decision_function(X)

        return _build_result("One-Class SVM", df, valid_idx, flag, -score, cols, label_cols, {"nu": nu})
