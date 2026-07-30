"""
train.py — Model Training Script

Trains all 4 ML models on the real insurance_variance_data_native.csv.
Saves trained models to ml/trained/.
Generates and displays the XGBoost confusion matrix.

Usage:
    python3 train.py
    python3 train.py --model xgboost     # train one model only
    python3 train.py --confusion-matrix  # show confusion matrix at end

Models trained:
  1. LightGBM   → ml/trained/lightgbm_underwriting_result_actual.pkl
  2. IsolationForest → ml/trained/isolation_forest.pkl
  3. XGBoost    → ml/trained/xgboost_classifier.pkl  + label_encoder
  4. KMeans     → ml/trained/kmeans.pkl + kmeans_scaler.pkl
  (Prophet is fit at query time — it uses the filtered time-series per query)
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from app.config import (
    DATASET_PATH, ML_MODEL_SAVE_DIR, LOG_LEVEL, LOG_FORMAT,
    LGBM_CFG, ISO_FOREST_CFG, XGBOOST_CFG, KMEANS_CFG,
    MODEL_REGISTRY_PATH,
)
from app.services.tools.sql_tool import SQLTool

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("ModelTrainer")

ML_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(obj, filename: str):
    path = ML_MODEL_SAVE_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"✅ Saved: {path}")
    return path


def _update_registry(key: str, metadata: dict):
    try:
        if MODEL_REGISTRY_PATH.exists():
            with open(MODEL_REGISTRY_PATH) as f:
                registry = json.load(f)
        else:
            registry = {"models": {}}
        registry["models"][key] = {**metadata, "timestamp": datetime.utcnow().isoformat()}
        with open(MODEL_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Registry update failed: {e}")


def _load_full_dataset() -> pd.DataFrame:
    """Load complete dataset via DuckDB."""
    sql = SQLTool()
    df = sql.conn.execute("SELECT * FROM insurance").df()
    logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: LightGBM — Multi-Feature Regression
# ══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(df: pd.DataFrame) -> dict:
    """Train LightGBM to predict underwriting_result_actual."""
    print("\n" + "═"*60)
    print("  MODEL 1: LightGBM — Multi-Feature Regression")
    print("═"*60)

    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        return {"error": "lightgbm / scikit-learn not installed"}

    target_col = "underwriting_result_actual"
    cat_cols = [c for c in LGBM_CFG.get("categorical_features", []) if c in df.columns]
    feature_cols = [c for c in LGBM_CFG.get("feature_columns", []) if c in df.columns]

    # Fallback: use both numeric columns and categorical columns as features
    if not feature_cols:
        num_cols = [c for c in df.columns
                    if c.endswith("_actual") and c != target_col
                    and pd.api.types.is_numeric_dtype(df[c])][:20]
        feature_cols = num_cols + cat_cols

    # Ensure all feature columns actually exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_m = df[[target_col] + feature_cols].copy()
    df_m[target_col] = pd.to_numeric(df_m[target_col], errors="coerce")
    df_m = df_m.dropna(subset=[target_col])

    for col in cat_cols:
        if col in df_m.columns:
            df_m[col] = df_m[col].astype("category")

    for col in [c for c in feature_cols if c not in cat_cols]:
        if col in df_m.columns and pd.api.types.is_numeric_dtype(df_m[col]):
            df_m[col] = df_m[col].fillna(df_m[col].median())

    X = df_m[feature_cols]
    y = df_m[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    hp = LGBM_CFG.get("hyperparameters", {})

    model = lgb.LGBMRegressor(
        n_estimators=hp.get("n_estimators", 200),
        learning_rate=hp.get("learning_rate", 0.05),
        num_leaves=hp.get("num_leaves", 31),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        verbose=-1,
        random_state=42,
    )
    fit_cat_cols = [c for c in cat_cols if c in X_train.columns]
    model.fit(X_train, y_train, categorical_feature=fit_cat_cols if fit_cat_cols else "auto")

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Feature importances
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=False).head(10)

    print(f"\n  Target    : {target_col}")
    print(f"  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  RMSE      : {rmse:,.2f} USD thousands")
    print("\n  Top 10 Feature Importances:")
    print(f"  {'Feature':<45} Importance")
    print("  " + "-"*55)
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] / fi_df["importance"].max() * 20)
        print(f"  {row['feature']:<45} {bar} {row['importance']:.0f}")

    _save(model, f"lightgbm_{target_col}.pkl")
    _update_registry("lightgbm_regressor", {
        "algorithm": "LightGBM", "target": target_col,
        "r2": round(r2, 4), "rmse": round(rmse, 2),
        "train_rows": len(X_train), "test_rows": len(X_test),
    })
    return {"r2": r2, "rmse": rmse}


# ══════════════════════════════════════════════════════════════════════════════
# Model 2: Isolation Forest — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(df: pd.DataFrame) -> dict:
    """Train Isolation Forest on ratio columns."""
    print("\n" + "═"*60)
    print("  MODEL 2: Isolation Forest — Anomaly Detection")
    print("═"*60)

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn not installed"}

    feature_cols = [c for c in ISO_FOREST_CFG.get("feature_columns", []) if c in df.columns]
    hp = ISO_FOREST_CFG.get("hyperparameters", {})
    contamination = hp.get("contamination", 0.05)

    df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_feat = df_feat.fillna(df_feat.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=hp.get("n_estimators", 200),
        random_state=42,
    )
    model.fit(X)

    labels = model.predict(X)
    scores = model.score_samples(X)
    n_anomalies = (labels == -1).sum()
    anomaly_rate = n_anomalies / len(labels) * 100

    print(f"\n  Features   : {len(feature_cols)} ratio columns")
    print(f"  Records    : {len(df_feat)}")
    print(f"  Anomalies  : {n_anomalies} ({anomaly_rate:.1f}%)")
    print(f"  Contamination set at: {contamination*100:.0f}%")
    print(f"\n  Most anomalous records (top 5 by score):")
    anomaly_df = df[feature_cols].copy()
    anomaly_df["score"] = scores
    anomaly_df["flag"]  = labels
    top5 = anomaly_df[anomaly_df["flag"] == -1].nsmallest(5, "score")
    if "loss_ratio_actual" in top5.columns and "combined_ratio_actual" in top5.columns:
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"    #{i}  loss_ratio={row.get('loss_ratio_actual', 'N/A'):.1f}%  "
                  f"combined_ratio={row.get('combined_ratio_actual', 'N/A'):.1f}%  "
                  f"score={row['score']:.4f}")

    _save(model, "isolation_forest.pkl")
    _save(scaler, "isolation_forest_scaler.pkl")
    _update_registry("isolation_forest", {
        "algorithm": "IsolationForest",
        "anomaly_count": int(n_anomalies),
        "anomaly_rate_pct": round(anomaly_rate, 2),
        "contamination": contamination,
        "feature_columns": feature_cols,
    })
    return {"anomaly_count": int(n_anomalies), "anomaly_rate_pct": anomaly_rate}


# ══════════════════════════════════════════════════════════════════════════════
# Model 3: XGBoost — Variance Driver Classification
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(df: pd.DataFrame, show_confusion_matrix: bool = True) -> dict:
    """Train XGBoost classifier to predict primary_variance_driver."""
    print("\n" + "═"*60)
    print("  MODEL 3: XGBoost — Variance Driver Classification")
    print("═"*60)

    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.metrics import (
            accuracy_score, f1_score,
            classification_report, confusion_matrix,
        )
    except ImportError:
        return {"error": "xgboost / scikit-learn not installed"}

    target_col   = XGBOOST_CFG.get("target_column", "primary_variance_driver")
    feature_cols = [c for c in XGBOOST_CFG.get("feature_columns", []) if c in df.columns]
    hp           = XGBOOST_CFG.get("hyperparameters", {})

    if target_col not in df.columns:
        print(f"  ❌ Target column '{target_col}' not found in dataset")
        return {"error": f"Column '{target_col}' not found"}

    df_m = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    for col in feature_cols:
        df_m[col] = pd.to_numeric(df_m[col], errors="coerce").fillna(0)

    # Classes with too few samples can't be learned or reliably evaluated,
    # and break a stratified split outright (a 1-sample class can't appear
    # in both train and test). Merge anything under MIN_CLASS_SIZE into a
    # single "OTHER" bucket rather than silently dragging macro-F1 down or
    # crashing.
    MIN_CLASS_SIZE = 20
    class_counts = df_m[target_col].value_counts()
    rare_classes = class_counts[class_counts < MIN_CLASS_SIZE].index.tolist()
    if rare_classes:
        print(f"\n  ⚠ Merging {len(rare_classes)} rare classes into 'OTHER' "
              f"(< {MIN_CLASS_SIZE} samples): {rare_classes}")
        df_m[target_col] = df_m[target_col].where(~df_m[target_col].isin(rare_classes), "OTHER")

    le = LabelEncoder()
    y = le.fit_transform(df_m[target_col].astype(str))
    X = df_m[feature_cols]
    classes = list(le.classes_)

    def _new_model():
        return xgb.XGBClassifier(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 5),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )

    # A single 80/20 split is too noisy to trust here — several classes
    # have under 30 total samples, so one split can land as few as 6-11
    # test examples for a class, swinging its F1 wildly by chance. 5-fold
    # stratified CV (mean across folds) is the trustworthy metric; the
    # single split below is kept only for the confusion matrix / feature
    # importances, which need one fixed train/test partition to display.
    cv_f1, cv_acc = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        cv_model = _new_model()
        cv_model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        cv_pred = cv_model.predict(X_te)
        cv_acc.append(accuracy_score(y_te, cv_pred))
        cv_f1.append(f1_score(y_te, cv_pred, average="macro", zero_division=0))
    cv_acc_mean, cv_acc_std = float(np.mean(cv_acc)), float(np.std(cv_acc))
    cv_f1_mean, cv_f1_std   = float(np.mean(cv_f1)), float(np.std(cv_f1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sample_weight = compute_sample_weight("balanced", y_train)

    model = _new_model()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  Target    : {target_col}")
    print(f"  Classes   : {len(classes)}")
    print(f"\n  5-fold CV (trustworthy estimate):")
    print(f"    Accuracy : {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"    F1 Macro : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"\n  Single 80/20 split (reference only — noisy on small classes):")
    print(f"    Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"    Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    F1 Macro : {f1:.4f}")

    # Feature importances
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(10)

    print("\n  Top 10 Feature Importances:")
    print(f"  {'Feature':<40} Importance")
    print("  " + "-"*50)
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] / fi_df["importance"].max() * 20)
        print(f"  {row['feature']:<40} {bar} {row['importance']:.4f}")

    # Classification Report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                labels=np.arange(len(classes)),
                                target_names=[c[:30] for c in classes],
                                zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    if show_confusion_matrix:
        _print_confusion_matrix(confusion_matrix(y_test, y_pred, labels=np.arange(len(classes))), classes)

    _save(model, "xgboost_classifier.pkl")
    _save(le,    "xgboost_label_encoder.pkl")
    _update_registry("xgboost_classifier", {
        "algorithm": "XGBoost", "target": target_col,
        # 5-fold CV mean is the headline metric — trustworthy given how few
        # samples some classes have. Single-split numbers kept for reference.
        "accuracy": round(cv_acc_mean, 4), "f1_macro": round(cv_f1_mean, 4),
        "accuracy_cv_std": round(cv_acc_std, 4), "f1_macro_cv_std": round(cv_f1_std, 4),
        "single_split_accuracy": round(accuracy, 4), "single_split_f1_macro": round(f1, 4),
        "classes": classes, "train_rows": len(X_train), "test_rows": len(X_test),
    })
    return {"accuracy": cv_acc_mean, "f1_macro": cv_f1_mean, "classes": classes}


def _print_confusion_matrix(cm: np.ndarray, classes: list[str]):
    """
    Print a terminal-friendly confusion matrix.
    Rows = Actual class. Columns = Predicted class.
    """
    print("\n" + "═"*60)
    print("  CONFUSION MATRIX — XGBoost Variance Driver Classifier")
    print("  Rows = Actual | Columns = Predicted")
    print("═"*60)

    # Shorten class names for display
    short = [c.replace("_variance", "").replace("_", " ")[:18] for c in classes]
    n = len(classes)
    col_w = 10

    # Header
    print("  Actual \\ Pred       ", end="")
    for s in short:
        print(f"{s:>{col_w}}", end="")
    print(f"  {'Total':>{col_w}}")
    print("  " + "-" * (22 + col_w * n + col_w + 2))

    row_totals  = cm.sum(axis=1)
    col_totals  = cm.sum(axis=0)
    grand_total = cm.sum()

    for i, actual in enumerate(short):
        print(f"  {actual:<22}", end="")
        for j in range(n):
            val = cm[i, j]
            marker = "◉" if i == j else " "   # Diagonal = correct predictions
            print(f"{marker}{val:>{col_w-1}}", end="")
        print(f"  {row_totals[i]:>{col_w}}")

    print("  " + "-" * (22 + col_w * n + col_w + 2))
    # Column totals
    print(f"  {'Total':<22}", end="")
    for t in col_totals:
        print(f"{t:>{col_w}}", end="")
    print(f"  {grand_total:>{col_w}}")

    # Per-class accuracy
    print("\n  Per-class Recall (diagonal / row total):")
    for i, cls in enumerate(short):
        recall = cm[i, i] / row_totals[i] if row_totals[i] > 0 else 0
        bar    = "█" * int(recall * 20)
        print(f"  {cls:<22} {bar:<22} {recall*100:.1f}%")

    print(f"\n  Overall Accuracy: {np.trace(cm) / grand_total * 100:.2f}%")
    print(f"  Total test records: {grand_total}")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════════
# Model 4: K-Means — Risk Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def train_kmeans(df: pd.DataFrame) -> dict:
    """Train K-Means for portfolio risk segmentation."""
    print("\n" + "═"*60)
    print("  MODEL 4: K-Means — Portfolio Risk Segmentation")
    print("═"*60)

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn not installed"}

    feature_cols = [c for c in KMEANS_CFG.get("feature_columns", []) if c in df.columns]
    n_clusters   = KMEANS_CFG.get("hyperparameters", {}).get("n_clusters", 3)
    risk_labels  = KMEANS_CFG.get("risk_labels", {}).get("by_combined_ratio", {})

    df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_feat = df_feat.fillna(df_feat.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Centroids in original scale
    centroids = scaler.inverse_transform(model.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=feature_cols)

    print(f"\n  Features  : {len(feature_cols)} ratio columns")
    print(f"  Records   : {len(df_feat)}")
    print(f"  Clusters  : {n_clusters}")

    low_t  = risk_labels.get("low_risk_threshold", 90)
    med_t  = risk_labels.get("medium_risk_threshold", 100)
    distribution = {i: int((labels == i).sum()) for i in range(n_clusters)}

    print("\n  Cluster Risk Profiles:")
    print(f"  {'Cluster':<10} {'Records':<10} {'Risk Label':<30} "
          f"{'Loss Ratio':<14} {'Combined Ratio'}")
    print("  " + "-"*80)
    for i in range(n_clusters):
        cr  = centroid_df.at[i, "combined_ratio_actual"] \
              if "combined_ratio_actual" in centroid_df.columns else 0
        lr  = centroid_df.at[i, "loss_ratio_actual"] \
              if "loss_ratio_actual" in centroid_df.columns else 0
        if cr < low_t:
            risk = "🟢 Low Risk (Profitable)"
        elif cr < med_t:
            risk = "🟡 Medium Risk (Near Breakeven)"
        else:
            risk = "🔴 High Risk (UWR Loss)"
        print(f"  {i:<10} {distribution[i]:<10} {risk:<35} {lr:.1f}%          {cr:.1f}%")

    _save(model,  "kmeans.pkl")
    _save(scaler, "kmeans_scaler.pkl")
    _update_registry("kmeans_segmentation", {
        "algorithm": "KMeans", "n_clusters": n_clusters,
        "distribution": distribution,
    })
    return {"n_clusters": n_clusters, "distribution": distribution}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train all ML models on the insurance dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["lightgbm", "isolation_forest", "xgboost", "kmeans", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        default=True,
        help="Show confusion matrix after XGBoost training (default: true)",
    )
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  Enterprise Insurance Analytics Agent — Model Trainer")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Save dir: {ML_MODEL_SAVE_DIR}")
    print("═"*60)

    df = _load_full_dataset()
    results = {}

    models_to_run = {
        "all":              ["lightgbm", "isolation_forest", "xgboost", "kmeans"],
        "lightgbm":         ["lightgbm"],
        "isolation_forest": ["isolation_forest"],
        "xgboost":          ["xgboost"],
        "kmeans":           ["kmeans"],
    }[args.model]

    if "lightgbm" in models_to_run:
        results["lightgbm"] = train_lightgbm(df)

    if "isolation_forest" in models_to_run:
        results["isolation_forest"] = train_isolation_forest(df)

    if "xgboost" in models_to_run:
        results["xgboost"] = train_xgboost(df, show_confusion_matrix=args.confusion_matrix)

    if "kmeans" in models_to_run:
        results["kmeans"] = train_kmeans(df)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TRAINING SUMMARY")
    print("═"*60)
    for name, result in results.items():
        if "error" in result:
            print(f"  ❌ {name:<20}: {result['error']}")
        else:
            metrics = "  |  ".join(f"{k}={v}" for k, v in result.items()
                                   if k not in ("classes", "distribution"))
            print(f"  ✅ {name:<20}: {metrics}")
    print(f"\n  All models saved to: {ML_MODEL_SAVE_DIR}/")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
