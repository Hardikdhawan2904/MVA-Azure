"""
ml/forecaster.py — Prophet + LightGBM Forecasting

Model 1: Prophet — Monthly time-series forecasting
  - Handles seasonality, missing months, uncertainty intervals
  - Used for: underwriting_result_actual, GWP, NEP

Model 2: LightGBM — Multi-feature regression
  - Used when multiple business drivers influence the target
  - Returns feature importances as business evidence
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from app.config import MODEL_REGISTRY_PATH, FORECAST_PERIODS
from app.services.ml.persistence import load_pickle

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Time-series forecasting using Meta's Prophet.

    Why Prophet?
    - Insurance data has quarterly seasonality (Q4 spikes for year-end renewals).
    - Prophet handles irregular time steps and missing months gracefully.
    - Built-in confidence intervals satisfy regulatory reporting requirements.
    - Produces decomposed trend + seasonality that maps to business understanding.
    """

    def __init__(self):
        self.model = None
        self.model_key = None

    def fit_and_forecast(
        self,
        ts_df: pd.DataFrame,
        periods: int = FORECAST_PERIODS,
        freq: str = "ME",
        yearly_seasonality: bool = True,
        quarterly_seasonality: bool = True,
    ) -> dict:
        """
        Fit Prophet to historical time series and generate forecast.

        Args:
            ts_df      : DataFrame with columns ['ds', 'y'] (date, value).
                         'ds' can be string or datetime; 'y' is numeric.
            periods    : Number of future periods to forecast (months).
            freq       : Pandas frequency string ('ME' = month-end).
            yearly_seasonality  : Add yearly seasonality component.
            quarterly_seasonality : Add quarterly seasonality (insurance peak).

        Returns:
            {
              "model": "Prophet",
              "historical_periods": int,
              "forecast_periods": int,
              "forecast": [{"ds": ..., "yhat": ..., "yhat_lower": ..., "yhat_upper": ...}],
              "trend_direction": "increasing" | "decreasing" | "flat",
              "components": {"trend": [...], "yearly": [...]},
            }
        """
        try:
            from prophet import Prophet
        except ImportError:
            return {"error": "prophet package not installed. Run: pip install prophet"}

        df = ts_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"]  = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

        if len(df) < 6:
            return {
                "error": f"Insufficient data for Prophet. Need >= 6 periods, got {len(df)}."
            }

        logger.info(f"Prophet: fitting on {len(df)} historical periods, "
                    f"forecasting {periods} periods ahead")

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=1000,
            seasonality_mode="multiplicative",
        )

        if quarterly_seasonality:
            model.add_seasonality(
                name="quarterly",
                period=91.25,
                fourier_order=5,
            )

        model.fit(df)
        self.model = model

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast_df = model.predict(future)

        # Separate historical fitted values from future forecast
        future_only = forecast_df.tail(periods)

        trend_start = forecast_df["trend"].iloc[0]
        trend_end   = forecast_df["trend"].iloc[-1]
        if trend_end > trend_start * 1.01:
            trend_direction = "increasing"
        elif trend_end < trend_start * 0.99:
            trend_direction = "decreasing"
        else:
            trend_direction = "flat"

        result = {
            "model":              "Prophet",
            "historical_periods": len(df),
            "forecast_periods":   periods,
            "last_actual":        round(df["y"].iloc[-1], 2),
            "last_actual_date":   str(df["ds"].iloc[-1].date()),
            "trend_direction":    trend_direction,
            "forecast": [
                {
                    "ds":         str(row["ds"].date()),
                    "yhat":       round(row["yhat"], 2),
                    "yhat_lower": round(row["yhat_lower"], 2),
                    "yhat_upper": round(row["yhat_upper"], 2),
                }
                for _, row in future_only.iterrows()
            ],
        }

        self._update_registry("prophet_forecast", {"algorithm": "Prophet", "periods": periods})
        logger.info(f"Prophet forecast complete: trend={trend_direction}, "
                    f"next_period_yhat={result['forecast'][0]['yhat']:,.2f}")
        return result


class LightGBMForecaster:
    """
    Multi-feature regression using LightGBM.

    Why LightGBM?
    - 141 mixed-type columns; handles categoricals natively (no one-hot encoding).
    - Fastest gradient boosting on tabular data at this row count.
    - Feature importance maps directly to business drivers.
    - Used when the forecast depends on multiple input variables (not just time).
    """

    def __init__(self):
        self.model = None
        self.feature_names = []

    def load_persisted(self, target_col: str) -> bool:
        """Load the train.py-fit LightGBM regressor for target_col. Returns
        False if no persisted model exists yet for that target."""
        model = load_pickle(f"lightgbm_{target_col}.pkl")
        if model is None:
            return False
        self.model = model
        # LightGBM's sklearn API stores the fitted feature names on the
        # model itself — no need to persist them separately.
        self.feature_names = list(getattr(model, "feature_name_", []))
        return True

    def get_key_drivers(self, target_col: str, top_n: int = 5) -> dict | None:
        """
        Report the persisted model's top feature importances for target_col
        — a static read of what the model learned, not a fresh fit or a
        live prediction. Returns None if no persisted model exists yet.
        """
        if self.model is None and not self.load_persisted(target_col):
            return None

        if not self.feature_names:
            return {"error": "Persisted model has no stored feature names."}

        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1

        return {
            "model":  "LightGBM",
            "source": "persisted",
            "target": target_col,
            "key_drivers": fi_df.to_dict(orient="records"),
        }

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        categorical_cols: list[str] | None = None,
        top_n_features: int = 10,
    ) -> dict:
        """
        Train LightGBM regressor and predict on the same dataset (in-sample).
        Returns predictions and feature importances for business evidence.

        Args:
            df             : Full DataFrame.
            target_col     : Column to predict (e.g. 'underwriting_result_actual').
            feature_cols   : Input feature columns.
            categorical_cols: Categorical feature columns (LightGBM handles natively).
            top_n_features : Number of top features to report.

        Returns:
            {
              "model": "LightGBM",
              "r2_score": float,
              "rmse": float,
              "feature_importances": [{"feature": ..., "importance": ..., "rank": ...}],
            }
        """
        try:
            import lightgbm as lgb
            from sklearn.metrics import r2_score, mean_squared_error
            from sklearn.model_selection import train_test_split
        except ImportError:
            return {"error": "lightgbm or scikit-learn not installed"}

        df_model = df[feature_cols + [target_col]].copy()
        df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")
        df_model = df_model.dropna(subset=[target_col])

        # Encode categoricals as category dtype (LightGBM requirement). NaN
        # must be filled BEFORE the category conversion, not after -- a
        # pandas Categorical can only be filled with a value that's already
        # one of its categories, so filling post-conversion with "Unknown"
        # (never one of the column's real values) raises "Cannot setitem on
        # a Categorical with a new category" unconditionally, regardless of
        # whether the column even has any NaN to fill.
        cat_cols = categorical_cols or []
        for col in cat_cols:
            if col in df_model.columns:
                df_model[col] = df_model[col].fillna("Unknown").astype("category")

        # Fill remaining NaN with median -- numeric feature columns not
        # already handled as categoricals above.
        for col in feature_cols:
            if col in df_model.columns and col not in cat_cols:
                if df_model[col].dtype.kind in "biufc":
                    df_model[col] = df_model[col].fillna(df_model[col].median())
                else:
                    df_model[col] = df_model[col].fillna("Unknown").astype("category")

        X = df_model[feature_cols]
        y = df_model[target_col]

        if len(X) < 20:
            return {"error": f"Insufficient rows for LightGBM training: {len(X)}"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"LightGBM: training on {len(X_train)} rows, target='{target_col}'")

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )

        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Feature importances
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n_features).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1

        self.model = model
        self.feature_names = feature_cols

        result = {
            "model":               "LightGBM",
            "target":              target_col,
            "train_rows":          len(X_train),
            "test_rows":           len(X_test),
            "r2_score":            round(r2, 4),
            "rmse":                round(rmse, 2),
            "feature_importances": fi_df.to_dict(orient="records"),
        }

        self._update_registry("lightgbm_regressor", {
            "algorithm": "LightGBM", "target": target_col, "r2": r2
        })
        logger.info(f"LightGBM fit complete: R²={r2:.4f}, RMSE={rmse:,.2f}")
        return result

    @staticmethod
    def _update_registry(key: str, metadata: dict):
        """Append model run metadata to model_registry.json."""
        try:
            with open(MODEL_REGISTRY_PATH, "r") as f:
                registry = json.load(f)
            registry["models"][key] = {**metadata, "timestamp": datetime.utcnow().isoformat()}
            with open(MODEL_REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update model registry: {e}")


# Attach registry update to ProphetForecaster
ProphetForecaster._update_registry = staticmethod(LightGBMForecaster._update_registry)

