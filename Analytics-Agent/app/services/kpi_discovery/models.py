"""app/services/kpi_discovery/models.py — Stage 2 of the Agent 3 redesign
(plan "zany-giggling-crayon"): the DiscoveredKPI shape and measure
categories.

Layer 1 (coarse): a metric column's semantic_type is classified into one
of these measure categories — "Revenue → Financial Measure" instead of
hardcoding column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FINANCIAL_MEASURE = "Financial Measure"
COMPENSATION_MEASURE = "Compensation Measure"
PAYMENT_MEASURE = "Payment Measure"
INSURANCE_MEASURE = "Insurance Measure"
DATE_DIMENSION = "Date Dimension"


@dataclass
class DiscoveredKPI:
    name: str
    formula: str                    # human-readable, e.g. "(revenue - cost) / revenue"
    source_columns: list[str]
    semantic_basis: str             # which rule/category combination fired
    category: str                   # one of the measure categories above
    kpi_type: str                   # "ratio" | "rate" | "sum" | "difference" | "distribution"
    target_columns: list[str] = field(default_factory=list)  # columns this KPI can be trended/compared on downstream
