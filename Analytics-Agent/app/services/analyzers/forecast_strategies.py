"""app/services/analyzers/forecast_strategies.py — Stage 7 "Improve Current
Fallbacks" (plan "zany-giggling-crayon"): deterministic forecast strategies
registered in config/model_registry.yml alongside Prophet/LightGBM.

Every strategy returns the same shape Prophet's ProphetForecaster.
fit_and_forecast() already produces (`forecast: [{"ds", "yhat",
"yhat_lower", "yhat_upper"}]`, `trend_direction`) so ForecastAnalyzer and
ExplanationTool's existing template-formatter code (which already reads
evidence["forecast"]) don't need an ML-vs-deterministic branch downstream
of the analyzer.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def _prepare_series(ts_df: pd.DataFrame) -> pd.DataFrame:
    df = ts_df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna().sort_values("ds").reset_index(drop=True)


def _inferred_step(df: pd.DataFrame) -> timedelta:
    if len(df) < 2:
        return timedelta(days=30)
    diffs = df["ds"].diff().dropna()
    return diffs.median() if not diffs.empty else timedelta(days=30)


def _trend_direction(values: list[float]) -> str:
    if len(values) < 2 or values[0] == 0:
        return "flat"
    change_pct = (values[-1] - values[0]) / abs(values[0]) * 100
    if change_pct > 1:
        return "increasing"
    if change_pct < -1:
        return "decreasing"
    return "flat"


class MovingAverageStrategy:
    """Forecasts every future period as the trailing window's mean —
    cheapest possible baseline, registered `min_rows: 3`."""

    def forecast(self, ts_df: pd.DataFrame, periods: int = 6, window: int = 3) -> dict:
        df = _prepare_series(ts_df)
        if len(df) < 3:
            return {"error": f"Insufficient data for Moving Average forecast: {len(df)} rows (need >= 3)"}

        step = _inferred_step(df)
        window = min(window, len(df))
        avg = df["y"].tail(window).mean()
        std = df["y"].tail(window).std(ddof=0) or 0.0
        last_date = df["ds"].iloc[-1]

        forecast = [
            {
                "ds": str((last_date + step * i).date()),
                "yhat": round(float(avg), 2),
                "yhat_lower": round(float(avg - std), 2),
                "yhat_upper": round(float(avg + std), 2),
            }
            for i in range(1, periods + 1)
        ]
        return {
            "evidence": {
                "model": "Moving Average",
                "historical_periods": len(df),
                "forecast_periods": periods,
                "last_actual": round(float(df["y"].iloc[-1]), 2),
                "last_actual_date": str(last_date.date()),
                "trend_direction": "flat",
                "forecast": forecast,
            },
            "metrics": {"window": window},
        }


class LinearTrendStrategy:
    """Ordinary least squares over the period index — the closest
    registry-driven equivalent to today's AnalyticsTool.trend() call."""

    def forecast(self, ts_df: pd.DataFrame, periods: int = 6) -> dict:
        df = _prepare_series(ts_df)
        if len(df) < 5:
            return {"error": f"Insufficient data for Linear Trend forecast: {len(df)} rows (need >= 5)"}

        step = _inferred_step(df)
        x = np.arange(len(df))
        slope, intercept = np.polyfit(x, df["y"].to_numpy(), 1)
        residuals = df["y"].to_numpy() - (slope * x + intercept)
        std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        last_date = df["ds"].iloc[-1]

        forecast = []
        for i in range(1, periods + 1):
            yhat = slope * (len(df) - 1 + i) + intercept
            forecast.append({
                "ds": str((last_date + step * i).date()),
                "yhat": round(float(yhat), 2),
                "yhat_lower": round(float(yhat - std), 2),
                "yhat_upper": round(float(yhat + std), 2),
            })

        return {
            "evidence": {
                "model": "Linear Trend",
                "historical_periods": len(df),
                "forecast_periods": periods,
                "last_actual": round(float(df["y"].iloc[-1]), 2),
                "last_actual_date": str(last_date.date()),
                "trend_direction": _trend_direction([float(df["y"].iloc[0]), float(df["y"].iloc[-1])]),
                "trend_slope": round(float(slope), 4),
                "forecast": forecast,
            },
            "metrics": {"residual_std": round(std, 4)},
        }


class ExponentialSmoothingStrategy:
    """Simple exponential smoothing (single parameter, no trend/seasonality
    component) — a middle ground between Moving Average's flat baseline
    and Linear Trend's straight-line extrapolation."""

    def forecast(self, ts_df: pd.DataFrame, periods: int = 6, alpha: float = 0.3) -> dict:
        df = _prepare_series(ts_df)
        if len(df) < 5:
            return {"error": f"Insufficient data for Exponential Smoothing forecast: {len(df)} rows (need >= 5)"}

        step = _inferred_step(df)
        values = df["y"].to_numpy()
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

        level = smoothed[-1]
        std = float(np.std(values - np.array(smoothed))) if len(values) > 1 else 0.0
        last_date = df["ds"].iloc[-1]

        forecast = [
            {
                "ds": str((last_date + step * i).date()),
                "yhat": round(float(level), 2),
                "yhat_lower": round(float(level - std), 2),
                "yhat_upper": round(float(level + std), 2),
            }
            for i in range(1, periods + 1)
        ]

        return {
            "evidence": {
                "model": "Exponential Smoothing",
                "historical_periods": len(df),
                "forecast_periods": periods,
                "last_actual": round(float(values[-1]), 2),
                "last_actual_date": str(last_date.date()),
                "trend_direction": _trend_direction(list(smoothed)),
                "smoothed_series": [round(float(v), 2) for v in smoothed],
                "forecast": forecast,
            },
            "metrics": {"alpha": alpha, "residual_std": round(std, 4)},
        }
