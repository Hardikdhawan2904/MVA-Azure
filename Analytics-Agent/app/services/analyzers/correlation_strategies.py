"""app/services/analyzers/correlation_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): Pearson/Spearman correlation strategies.

Both share the same interface — `.compute(df, columns)` — differing only
in which SciPy correlation function they call per column pair.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy import stats


class _CorrelationStrategyBase:
    _method_name = ""
    _scipy_fn = None

    def compute(self, df: pd.DataFrame, columns: list[str]) -> dict:
        cols = [c for c in columns if c in df.columns]
        if len(cols) < 2:
            return {"error": f"{self._method_name} correlation needs >= 2 numeric columns, found {len(cols)}"}

        numeric = df[cols].apply(pd.to_numeric, errors="coerce")

        matrix: dict[str, dict[str, float | None]] = {c: {} for c in cols}
        p_values: dict[str, dict[str, float | None]] = {c: {} for c in cols}
        pairs = []

        for a in cols:
            matrix[a][a] = 1.0
            p_values[a][a] = 0.0

        for a, b in combinations(cols, 2):
            paired = numeric[[a, b]].dropna()
            if len(paired) < 3:
                matrix[a][b] = matrix[b][a] = None
                p_values[a][b] = p_values[b][a] = None
                continue
            corr, p_value = self._scipy_fn(paired[a], paired[b])
            corr, p_value = round(float(corr), 4), round(float(p_value), 4)
            matrix[a][b] = matrix[b][a] = corr
            p_values[a][b] = p_values[b][a] = p_value
            pairs.append({"column_a": a, "column_b": b, "correlation": corr, "p_value": p_value})

        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

        return {
            "evidence": {
                "method": self._method_name,
                "correlation_matrix": matrix,
                "p_values": p_values,
                "pairs": pairs,
                "strongest_pair": pairs[0] if pairs else None,
            },
            "metrics": {"columns_analysed": len(cols), "pairs_analysed": len(pairs)},
        }


class PearsonStrategy(_CorrelationStrategyBase):
    _method_name = "Pearson"
    _scipy_fn = staticmethod(stats.pearsonr)


class SpearmanStrategy(_CorrelationStrategyBase):
    _method_name = "Spearman"
    _scipy_fn = staticmethod(stats.spearmanr)
