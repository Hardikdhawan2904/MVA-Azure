"""app/services/analyzers/association_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): association_rules strategies registered in
config/model_registry.yml.

Apriori is registered but `enabled: false` (needs the mlxtend package —
Phase 6, optional); ContingencyTableStrategy is the always-available
pandas-only default the registry's own comment describes.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd


class ContingencyTableStrategy:
    """Pairwise co-occurrence via pandas crosstab — no external
    dependency, works for any pair of categorical dimensions."""

    def compute(self, df: pd.DataFrame, dimension_cols: list[str]) -> dict:
        cols = [c for c in dimension_cols if c in df.columns]
        if len(cols) < 2:
            return {"error": f"Contingency Table needs >= 2 categorical columns, found {len(cols)}"}

        results = []
        for a, b in combinations(cols, 2):
            table = pd.crosstab(df[a], df[b])
            if table.empty:
                continue
            conditional = (table.T / table.sum(axis=1)).T.round(4)
            results.append({
                "column_a": a,
                "column_b": b,
                "co_occurrence_matrix": table.to_dict(),
                "conditional_probabilities": conditional.to_dict(),
            })

        if not results:
            return {"error": "No usable categorical column pairs for Contingency Table analysis."}

        return {
            "evidence": {
                "model": "Contingency Table Co-occurrence",
                "pairs_analysed": len(results),
                "results": results,
            }
        }
