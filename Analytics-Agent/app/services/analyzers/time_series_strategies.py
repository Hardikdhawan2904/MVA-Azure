"""app/services/analyzers/time_series_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): the sole "time_series_analysis" purpose strategy
registered in config/model_registry.yml. Descriptive (not predictive —
that's the "forecast" purpose's job), always deterministic by design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TimeSeriesDescriptiveStrategy:
    def compute(self, ts_df: pd.DataFrame) -> dict:
        df = ts_df[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna().sort_values("ds").reset_index(drop=True)

        if len(df) < 2:
            return {"error": f"Insufficient data for time-series analysis: {len(df)} rows (need >= 2)"}

        values = df["y"].to_numpy()
        volatility = float(np.std(values)) / abs(float(np.mean(values))) if np.mean(values) != 0 else None

        # Naive seasonality signal: autocorrelation at a 4-period lag (a
        # quarter, for quarterly/monthly-ish series) noticeably above what
        # white noise would produce — a hint, not a statistical test.
        seasonality_flag = False
        if len(values) >= 8:
            lag = 4
            autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            seasonality_flag = bool(not np.isnan(autocorr) and autocorr > 0.5)

        return {
            "evidence": {
                "model": "Time Series Descriptive",
                "data_points": len(df),
                "start_date": str(df["ds"].iloc[0].date()),
                "end_date": str(df["ds"].iloc[-1].date()),
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "volatility_ratio": round(volatility, 4) if volatility is not None else None,
                "seasonality_flag": seasonality_flag,
                "series_summary": f"{len(df)} periods from {df['ds'].iloc[0].date()} to {df['ds'].iloc[-1].date()}",
            }
        }
