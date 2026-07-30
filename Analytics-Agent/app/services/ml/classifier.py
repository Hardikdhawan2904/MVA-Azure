"""
ml/classifier.py — XGBoost Variance Classifier + K-Means Segmentation

Model 4: XGBoost Classifier — Variance Driver Classification
  - Predicts primary_variance_driver from variance columns
  - SHAP values provide per-prediction explainability

Model 5: K-Means — Risk Profile Segmentation
  - Unsupervised clustering on KPI ratios
  - Cluster centroids = interpretable risk tiers
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from app.config import MODEL_REGISTRY_PATH
from app.services.ml.persistence import load_pickle

logger = logging.getLogger(__name__)

VARIANCE_FEATURE_COLS = [
    "exposure_growth_variance",
    "premium_rate_variance",
    "policy_mix_variance",
    "claim_frequency_variance",
    "claim_severity_variance",
    "catastrophe_loss_variance",
    "fraud_leakage_variance",
    "premium_nonpayment_variance",
    "fx_translation_variance",
    "reinsurance_recovery_variance",
    "reserve_development_variance",
    "acquisition_cost_variance",
    "claims_processing_delay_variance",
    "one_off_variance_amount",
    "variance_vs_budget_amount",
    "variance_vs_prior_year_amount",
]

SEGMENT_FEATURE_COLS = [
    "loss_ratio_actual",
    "expense_ratio_actual",
    "combined_ratio_actual",
    "claim_frequency_actual",
    "average_claim_severity_actual",
    "renewal_rate_actual",
    "policy_lapse_rate_actual",
    "reserve_adequacy_ratio_actual",
]


class VarianceClassifier:
    """
    XGBoost multi-class classifier to predict the primary variance driver.

    Why XGBoost?
    - primary_variance_driver is pre-labelled → supervised classification.
    - XGBoost is the gold standard for multi-class tabular classification.
    - SHAP values give per-prediction feature attributions — no hallucination.
    - Predictions are evidence-backed by feature weights.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = []
        self.classes_ = []

    def load_persisted(self) -> bool:
        """Load the train.py-fit model + label encoder. Returns False (and
        leaves self.model unset) if no persisted artifacts exist yet."""
        model = load_pickle("xgboost_classifier.pkl")
        label_encoder = load_pickle("xgboost_label_encoder.pkl")
        if model is None or label_encoder is None:
            return False
        self.model = model
        self.label_encoder = label_encoder
        # Read the actual columns the booster was fit on, not our own guess
        # at what train.py used — this is the source of truth XGBoost itself
        # validates predict() calls against.
        booster_features = model.get_booster().feature_names
        self.feature_cols = list(booster_features) if booster_features else list(VARIANCE_FEATURE_COLS)
        self.classes_ = list(label_encoder.classes_)
        return True

    def predict_batch_with_explanation(self, df: pd.DataFrame, top_n: int = 5) -> dict | None:
        """
        Predict the primary variance driver for every row in df using the
        persisted model, plus SHAP-based feature attribution aggregated
        across those rows. Returns None if no persisted model exists yet,
        so the caller can skip this evidence entirely on a cold start.
        """
        if self.model is None and not self.load_persisted():
            return None

        try:
            import shap
        except ImportError:
            return {"error": "shap not installed. Run: pip install shap"}

        # Unlike fit_and_evaluate() (which trains on whatever's available),
        # a persisted model must see exactly the columns it was fit on, in
        # the same order — narrowing to a subset would either error out or
        # silently feed it the wrong feature vector. If this query's data
        # is missing any, skip this evidence rather than guess.
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            return {"error": f"Persisted classifier expects columns missing from this "
                              f"query's data: {missing}"}

        X = df[self.feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        if X.empty:
            return {"error": "No rows to classify after filtering."}

        pred_encoded = self.model.predict(X)
        pred_labels = self.label_encoder.inverse_transform(pred_encoded)

        # Majority vote across the filtered rows is the "predicted driver"
        # for this query; per-row predictions are kept for confidence.
        pred_counts = pd.Series(pred_labels).value_counts()
        majority_driver = pred_counts.index[0]
        confidence = round(float(pred_counts.iloc[0] / len(pred_labels)), 4)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        class_idx = list(self.label_encoder.classes_).index(majority_driver) \
            if majority_driver in self.label_encoder.classes_ else 0
        sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values[:, :, class_idx]
        mean_abs_shap = np.abs(sv).mean(axis=0)

        top_features = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        return {
            "predicted_driver": majority_driver,
            "confidence": confidence,
            "rows_classified": len(pred_labels),
            "top_features": top_features.round(4).to_dict(orient="records"),
        }

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "primary_variance_driver",
        feature_cols: list[str] | None = None,
    ) -> dict:
        """
        Train XGBoost on pre-labelled variance data and evaluate.

        Args:
            df         : Full DataFrame with variance columns + target.
            target_col : The classification target (default: primary_variance_driver).
            feature_cols: Features to train on (defaults to VARIANCE_FEATURE_COLS).

        Returns:
            {
              "model": "XGBoost",
              "target": str,
              "accuracy": float,
              "classes": [...],
              "feature_importances": [...],
            }
        """
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_sample_weight
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError:
            return {"error": "xgboost or scikit-learn not installed"}

        cols = feature_cols or [c for c in VARIANCE_FEATURE_COLS if c in df.columns]

        df_model = df[cols + [target_col]].copy()
        df_model = df_model.dropna(subset=[target_col])
        df_model = df_model.dropna(subset=cols, how="all")

        for col in cols:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

        if len(df_model) < 20:
            return {"error": f"Insufficient labelled rows: {len(df_model)} (need >= 20)"}

        # Classes with too few samples break a stratified split outright (a
        # 1-sample class can't appear in both train and test) and can't be
        # learned or evaluated reliably anyway — merge into "OTHER".
        MIN_CLASS_SIZE = 20
        class_counts = df_model[target_col].value_counts()
        rare_classes = class_counts[class_counts < MIN_CLASS_SIZE].index.tolist()
        if rare_classes:
            logger.info(f"Merging {len(rare_classes)} rare classes into 'OTHER': {rare_classes}")
            df_model[target_col] = df_model[target_col].where(
                ~df_model[target_col].isin(rare_classes), "OTHER"
            )

        le = LabelEncoder()
        y = le.fit_transform(df_model[target_col].astype(str))
        X = df_model[cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        sample_weight = compute_sample_weight("balanced", y_train)

        logger.info(f"XGBoost: training on {len(X_train)} rows, "
                    f"{len(le.classes_)} classes, target='{target_col}'")

        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Feature importances
        fi = pd.DataFrame({
            "feature":    cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10).reset_index(drop=True)
        fi["rank"] = fi.index + 1

        self.model         = model
        self.label_encoder = le
        self.feature_cols  = cols
        self.classes_      = list(le.classes_)

        result = {
            "model":               "XGBoost",
            "target":              target_col,
            "accuracy":            round(accuracy, 4),
            "f1_macro":            round(f1_macro, 4),
            "classes":             self.classes_,
            "train_rows":          len(X_train),
            "test_rows":           len(X_test),
            "feature_importances": fi.to_dict(orient="records"),
        }

        self._update_registry("xgboost_classifier", {
            "algorithm": "XGBoost", "accuracy": accuracy, "f1_macro": f1_macro, "target": target_col
        })
        logger.info(f"XGBoost fit complete: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")
        return result

    def predict_with_explanation(self, record: dict) -> dict:
        """
        Predict the primary variance driver for a new record.
        Returns prediction + SHAP-based explanation.
        """
        if not self.model:
            return {"error": "Model not trained yet. Call fit_and_evaluate() first."}

        try:
            import shap
        except ImportError:
            return {"error": "shap not installed. Run: pip install shap"}

        X_new = pd.DataFrame([record])[self.feature_cols]
        X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(0)

        pred_encoded = self.model.predict(X_new)[0]
        pred_label   = self.label_encoder.inverse_transform([pred_encoded])[0]

        # SHAP explanation
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_new)

        # For multi-class, pick SHAP for predicted class
        class_idx   = list(self.label_encoder.classes_).index(pred_label) \
                      if pred_label in self.label_encoder.classes_ else 0
        sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]

        shap_df = pd.DataFrame({
            "feature": self.feature_cols,
            "shap_value": sv,
        }).sort_values("shap_value", key=abs, ascending=False).head(5)

        return {
            "predicted_driver": pred_label,
            "shap_explanation": shap_df.to_dict(orient="records"),
        }

    @staticmethod
    def _update_registry(key: str, metadata: dict):
        try:
            with open(MODEL_REGISTRY_PATH, "r") as f:
                registry = json.load(f)
            registry["models"][key] = {**metadata, "timestamp": datetime.utcnow().isoformat()}
            with open(MODEL_REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Registry update failed: {e}")


class RiskSegmenter:
    """
    K-Means clustering for insurance risk profile segmentation.

    Why K-Means?
    - Unsupervised — no labels needed.
    - Cluster centroids = interpretable risk profiles (Low / Medium / High risk).
    - Applied to KPI ratio columns for portfolio risk analysis.
    - Invoked only for segmentation queries.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.n_clusters = 3
        self.feature_cols = []

    def assign(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
    ) -> dict | None:
        """
        Assign records to the persisted (train.py-fit) KMeans centroids —
        no re-clustering here. Returns None if no persisted model exists,
        so the caller can fall back to segment() (fit-fresh) on a cold start.

        This is the fix for a real correctness bug: segment() re-clusters
        from scratch on whatever rows a query's filters returned, so cluster
        0 in one query and cluster 0 in another aren't the same risk profile
        — the labels are arbitrary per fit. Assigning against one fixed set
        of centroids makes "risk segment" mean the same thing every time.
        """
        model = load_pickle("kmeans.pkl")
        scaler = load_pickle("kmeans_scaler.pkl")
        if model is None or scaler is None:
            return None

        # Same reasoning as AnomalyDetector.score(): the scaler must see
        # exactly the columns it was fit on. Fall back to fit-fresh (which
        # adapts to whatever's available) rather than erroring if this
        # query's data is missing one of them.
        cols = list(getattr(scaler, "feature_names_in_", feature_cols or SEGMENT_FEATURE_COLS))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.info(f"Persisted KMeans expects columns missing from this query's "
                        f"data ({missing}) — falling back to fit-fresh.")
            return None

        df_feat = df[cols].copy().apply(pd.to_numeric, errors="coerce")
        df_feat = df_feat.fillna(df_feat.median())
        valid_idx = df_feat.index

        if df_feat.empty:
            return {"error": "No rows to segment after filtering."}

        X = scaler.transform(df_feat)
        labels = model.predict(X)

        result_df = df.loc[valid_idx].copy()
        result_df["cluster"] = labels

        centroids_orig = scaler.inverse_transform(model.cluster_centers_)
        centroid_df = pd.DataFrame(centroids_orig, columns=cols)
        centroid_df.index.name = "cluster"
        centroid_df = centroid_df.reset_index()

        if "combined_ratio_actual" in cols:
            cr_idx = cols.index("combined_ratio_actual")
            centroid_df["risk_label"] = centroid_df.apply(
                lambda row: self._risk_label(centroids_orig[int(row["cluster"])][cr_idx]),
                axis=1,
            )
        else:
            centroid_df["risk_label"] = [f"Segment {i+1}" for i in range(len(centroid_df))]

        distribution = result_df["cluster"].value_counts().to_dict()
        output_cols = list(label_cols or []) + ["cluster"]
        output_records = result_df[[c for c in output_cols if c in result_df.columns]].to_dict(orient="records")

        return {
            "model":                 "KMeans",
            "source":                "persisted",
            "n_clusters":            len(centroid_df),
            "feature_columns":       cols,
            "cluster_profiles":      centroid_df.round(2).to_dict(orient="records"),
            "segment_distribution":  {int(k): int(v) for k, v in distribution.items()},
            "records_with_segments": output_records,
        }

    def segment(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        n_clusters: int = 3,
        label_cols: list[str] | None = None,
    ) -> dict:
        """
        Cluster records into risk segments using K-Means.

        Args:
            df           : DataFrame with ratio columns.
            feature_cols : Features to cluster on (defaults to SEGMENT_FEATURE_COLS).
            n_clusters   : Number of clusters (default 3 = Low/Medium/High).
            label_cols   : Columns to include in output for identification.

        Returns:
            {
              "model": "KMeans",
              "n_clusters": int,
              "cluster_profiles": [...],
              "segment_distribution": {cluster_id: count},
              "records_with_segments": [{"cluster": ..., <label_cols>..., <features>...}],
            }
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "scikit-learn not installed"}

        cols = feature_cols or [c for c in SEGMENT_FEATURE_COLS if c in df.columns]
        if not cols:
            return {"error": "No ratio columns available for segmentation"}

        df_feat = df[cols].copy().apply(pd.to_numeric, errors="coerce")
        df_feat = df_feat.fillna(df_feat.median())
        valid_idx = df_feat.index

        if len(df_feat) < n_clusters * 2:
            return {"error": f"Too few records ({len(df_feat)}) for {n_clusters} clusters"}

        logger.info(f"KMeans: clustering {len(df_feat)} records into {n_clusters} clusters")

        scaler = StandardScaler()
        X = scaler.fit_transform(df_feat)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        self.model      = model
        self.scaler     = scaler
        self.n_clusters = n_clusters
        self.feature_cols = cols

        result_df = df.loc[valid_idx].copy()
        result_df["cluster"] = labels

        # Cluster centroids (in original scale)
        centroids_scaled = model.cluster_centers_
        centroids_orig   = scaler.inverse_transform(centroids_scaled)
        centroid_df = pd.DataFrame(centroids_orig, columns=cols)
        centroid_df.index.name = "cluster"
        centroid_df = centroid_df.reset_index()

        # Name clusters by combined ratio (proxy for risk)
        if "combined_ratio_actual" in cols:
            cr_idx = cols.index("combined_ratio_actual")
            centroid_df["risk_label"] = centroid_df.apply(
                lambda row: self._risk_label(centroids_orig[int(row["cluster"])][cr_idx]),
                axis=1,
            )
        else:
            centroid_df["risk_label"] = [f"Segment {i+1}" for i in range(n_clusters)]

        distribution = result_df["cluster"].value_counts().to_dict()

        # Build output
        output_cols = list(label_cols or []) + ["cluster"]
        output_records = result_df[
            [c for c in output_cols if c in result_df.columns]
        ].to_dict(orient="records")

        result = {
            "model":              "KMeans",
            "n_clusters":         n_clusters,
            "feature_columns":    cols,
            "cluster_profiles":   centroid_df.round(2).to_dict(orient="records"),
            "segment_distribution": {int(k): int(v) for k, v in distribution.items()},
            "records_with_segments": output_records,
        }

        VarianceClassifier._update_registry("kmeans_segmentation", {
            "algorithm": "KMeans", "n_clusters": n_clusters
        })
        logger.info(f"KMeans complete: {n_clusters} segments, "
                    f"sizes={dict(sorted(distribution.items()))}")
        return result

    @staticmethod
    def _risk_label(combined_ratio: float) -> str:
        if combined_ratio < 90:
            return "Low Risk (Profitable)"
        elif combined_ratio < 100:
            return "Medium Risk (Near Breakeven)"
        else:
            return "High Risk (Underwriting Loss)"
