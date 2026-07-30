"""app/services/analyzers/classification_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): classification strategies registered in
config/model_registry.yml alongside XGBoost Classifier (VarianceClassifier).
"""

from __future__ import annotations

import pandas as pd


class RandomForestClassifierStrategy:
    """scikit-learn RandomForestClassifier — same evidence shape as
    VarianceClassifier.fit_and_evaluate() (accuracy/classes/feature
    importances) so ClassificationAnalyzer treats both identically."""

    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        cols = [c for c in (feature_cols or []) if c in df.columns and c != target_col]
        if not cols or target_col not in df.columns:
            return {"error": f"Classification target/feature columns not found: target={target_col!r}, features={cols}"}

        df_model = df[cols + [target_col]].copy().dropna(subset=[target_col])
        df_model = pd.get_dummies(df_model, columns=[c for c in cols if df_model[c].dtype == object])
        encoded_cols = [c for c in df_model.columns if c != target_col]
        for col in encoded_cols:
            if df_model[col].dtype.kind in "biufc":
                df_model[col] = df_model[col].fillna(df_model[col].median())

        if len(df_model) < 30:
            return {"error": f"Insufficient rows for Random Forest Classifier: {len(df_model)} (need >= 30)"}

        le = LabelEncoder()
        y = le.fit_transform(df_model[target_col].astype(str))
        X = df_model[encoded_cols]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fi = pd.DataFrame({
            "feature": encoded_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10).reset_index(drop=True)
        fi["rank"] = fi.index + 1

        return {
            "model": "Random Forest Classifier",
            "target": target_col,
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "classes": list(le.classes_),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_importances": fi.to_dict(orient="records"),
        }


class LogisticRegressionStrategy:
    """scikit-learn LogisticRegression — the cheap ML tier (cost_tier:
    cheap), numeric-only per the registry (supports_categorical: false)."""

    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        cols = [c for c in (feature_cols or []) if c in df.columns and c != target_col]
        if not cols or target_col not in df.columns:
            return {"error": f"Classification target/feature columns not found: target={target_col!r}, features={cols}"}

        df_model = df[cols + [target_col]].copy()
        for col in cols:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
        df_model = df_model.dropna(subset=[target_col])
        df_model[cols] = df_model[cols].fillna(df_model[cols].median())

        if len(df_model) < 20:
            return {"error": f"Insufficient rows for Logistic Regression: {len(df_model)} (need >= 20)"}

        le = LabelEncoder()
        y = le.fit_transform(df_model[target_col].astype(str))
        X = StandardScaler().fit_transform(df_model[cols])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        return {
            "model": "Logistic Regression",
            "target": target_col,
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "classes": list(le.classes_),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }


class MajorityClassBaselineStrategy:
    """Always predicts the most frequent class — the deterministic
    (requires_ml: false) baseline every classification purpose falls back
    to when nothing else is viable (Null Object guarantee, Stage 6)."""

    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> dict:
        if target_col not in df.columns:
            return {"error": f"Classification target column not found: {target_col!r}"}

        target = df[target_col].dropna()
        if target.empty:
            return {"error": f"No non-null values in target column '{target_col}'."}

        counts = target.value_counts()
        majority_class = counts.index[0]
        accuracy = round(float(counts.iloc[0] / len(target)), 4)

        return {
            "model": "Majority Class Baseline",
            "target": target_col,
            "predicted_class": str(majority_class),
            "accuracy": accuracy,
            "classes": [str(c) for c in counts.index.tolist()],
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
        }
