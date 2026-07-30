"""app/services/analyzers/regression_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): regression strategies registered in
config/model_registry.yml alongside LightGBM Regressor.

CatBoost Regressor is registered but `enabled: false` (needs the catboost
package — Phase 6, optional) so ModelSelector never selects it; no
implementation class exists for it yet, matching ARIMA/Apriori's same
deferred-dependency treatment elsewhere in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomForestRegressorStrategy:
    """scikit-learn RandomForestRegressor — a non-boosted ML alternative
    to LightGBM, same evidence shape (train/test split, R2, RMSE, feature
    importances) so RegressionAnalyzer treats both identically."""

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        categorical_cols: list[str] | None = None,
        top_n_features: int = 10,
    ) -> dict:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        cols = [c for c in feature_cols if c in df.columns]
        if not cols or target_col not in df.columns:
            return {"error": f"Regression target/feature columns not found: target={target_col!r}, features={cols}"}

        df_model = df[cols + [target_col]].copy()
        df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")
        df_model = df_model.dropna(subset=[target_col])

        cat_cols = [c for c in (categorical_cols or []) if c in df_model.columns]
        df_model = pd.get_dummies(df_model, columns=cat_cols, dummy_na=False)
        encoded_feature_cols = [c for c in df_model.columns if c != target_col]

        for col in encoded_feature_cols:
            if df_model[col].dtype.kind in "biufc":
                df_model[col] = df_model[col].fillna(df_model[col].median())

        X, y = df_model[encoded_feature_cols], df_model[target_col]
        if len(X) < 30:
            return {"error": f"Insufficient rows for Random Forest Regressor: {len(X)} (need >= 30)"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        fi_df = pd.DataFrame({
            "feature": encoded_feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n_features).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1

        return {
            "model": "Random Forest Regressor",
            "target": target_col,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "r2_score": round(float(r2), 4),
            "rmse": round(rmse, 2),
            "feature_importances": fi_df.to_dict(orient="records"),
        }


class LinearRegressionStrategy:
    """Plain OLS, in-sample only — the deterministic (requires_ml: false)
    regression fallback: cheap, closed-form, no train/test split needed
    to be trustworthy at this tier."""

    def fit_and_predict(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols or target_col not in df.columns:
            return {"error": f"Regression target/feature columns not found: target={target_col!r}, features={cols}"}

        data = df[cols + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 5:
            return {"error": f"Insufficient rows for Linear Regression: {len(data)} (need >= 5)"}

        X = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
        y = data[target_col].to_numpy()
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

        y_pred = X @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "model": "Linear Regression",
            "target": target_col,
            "rows_used": len(data),
            "r2_score": round(float(r2), 4),
            "intercept": round(float(coeffs[0]), 4),
            "coefficients": [{"feature": c, "coefficient": round(float(coeffs[i + 1]), 4)} for i, c in enumerate(cols)],
        }
