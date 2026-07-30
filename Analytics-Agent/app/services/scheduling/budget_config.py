"""app/services/scheduling/budget_config.py — Stage 5's BudgetConfig,
loaded from config/scheduling_budget.yml (app.config.SCHEDULING_BUDGET).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import SCHEDULING_BUDGET


@dataclass
class BudgetConfig:
    max_parallel_analyses: int = 8
    max_ml_analyses: int = 3
    max_expensive_operations: int = 2


def load_budget_config() -> BudgetConfig:
    return BudgetConfig(
        max_parallel_analyses=SCHEDULING_BUDGET.get("max_parallel_analyses", 8),
        max_ml_analyses=SCHEDULING_BUDGET.get("max_ml_analyses", 3),
        max_expensive_operations=SCHEDULING_BUDGET.get("max_expensive_operations", 2),
    )
