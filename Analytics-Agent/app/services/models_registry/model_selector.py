"""app/services/models_registry/model_selector.py — Stage 6 of the Agent 3
redesign (plan "zany-giggling-crayon"): ModelSelector (Strategy pattern).

Never a fixed intent -> algorithm dict (replaces today's `_ML_GATED_
ENGINES = {"forecast": "Prophet", ...}` literal in graph.py) — entirely
registry-driven. `ml_execution_allowed AND capability_profile`'s execution
gate together decide ML vs. deterministic; requirements filtering (min
rows, needs a datetime column, needs a target, ...) picks among the
survivors; cost_tier + the remaining expensive-operation budget breaks
further ties, cheapest-first once the budget is exhausted.

Stateful per instance (mirrors this codebase's existing convention —
MLTool/ExplanationTool/MemoryManager are all request-scoped stateful
objects too): construct one ModelSelector per request, `select()` is
called once per scheduled analysis in sequence, and the expensive-
operation counter accumulates across those calls within that request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.capability_resolution.models import CapabilityProfile, capability_for_analysis_type
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.algorithm_spec import COST_EXPENSIVE, AlgorithmSpec
from app.services.models_registry.model_registry import ModelRegistry
from app.services.scheduling.budget_config import BudgetConfig
from app.services.scheduling.models import ScheduledAnalysis


@dataclass
class SelectedModel:
    algorithm: str | None
    implementation_class: str | None
    requires_ml: bool
    cost_tier: str | None
    reasons: list[str] = field(default_factory=list)


class ModelSelector:
    def __init__(self, registry: ModelRegistry, budget: BudgetConfig):
        self._registry = registry
        self._budget = budget
        self._expensive_used = 0

    def select(
        self,
        scheduled_analysis: ScheduledAnalysis,
        dataset_context: DatasetContext,
        capability_profile: CapabilityProfile,
        preferred_algorithm: str | None = None,
    ) -> SelectedModel:
        """preferred_algorithm (Phase 4, plan "zany-giggling-crayon"): a
        domain plugin's get_preferred_deterministic_strategy(analysis_type)
        result, passed in by the caller — ModelSelector stays decoupled
        from DomainPlugin itself (duck-typed string in, same boundary
        discipline as every other "consume an already-computed value,
        don't reach into another layer" seam in this codebase). Only
        reorders the deterministic candidate list (never the ML one) —
        this is what keeps a plugin's existing narrative/output byte-
        identical even as the registry's own default priority order
        evolves; see get_preferred_deterministic_strategy()'s docstring.
        """
        analysis_type = scheduled_analysis.planned_analysis.analysis_type
        reasons: list[str] = []

        ml_allowed = (
            scheduled_analysis.ml_execution_allowed
            and capability_profile.is_analysis_type_ml_viable(analysis_type)
        )
        capability = capability_for_analysis_type(analysis_type)
        if capability is None:
            # No readiness gate exists for this analysis type at all (trend,
            # root_cause, ...) — deterministic-first by design, not because
            # any threshold check failed. Must never be confused with the
            # branch below: a caller (graph.py's trace builder) that saw
            # "execution gate did not" here would build a gate object
            # showing the CURRENT ml_readiness_score against the threshold
            # with a hardcoded passed=False, even when the score is
            # genuinely above threshold — self-contradictory and wrong,
            # since no such gate was ever evaluated for this analysis type.
            reasons.append(f"'{analysis_type}' has no ML-capable algorithm registered — deterministic by design")
        elif scheduled_analysis.ml_execution_allowed and not capability_profile.is_analysis_type_ml_viable(analysis_type):
            # Surface capability_profile's own execution.reason (Stage 1's
            # "ML readiness (X%) below the Y% threshold..." text) rather
            # than just naming the gate — gives the deterministic-fallback
            # audit trail the same readiness-score detail today's old
            # handlers' bespoke fallback_reason strings used to include.
            execution = capability_profile.capabilities.get(capability).execution
            detail = f" ({execution.reason})" if execution and execution.reason else ""
            reasons.append(f"Scheduler allowed ML, but capability resolution's execution gate did not{detail} — deterministic only")
        elif not scheduled_analysis.ml_execution_allowed:
            reasons.append("Scheduler budget did not allocate an ML slot to this analysis — deterministic only")
        else:
            reasons.append("ML execution allowed (scheduler budget + capability resolution both passed)")

        target_columns = scheduled_analysis.planned_analysis.target_columns
        candidates = self._registry.ml_algorithms_for(analysis_type) if ml_allowed else []
        candidates = self._filter_by_requirements(candidates, dataset_context, target_columns)

        if not candidates:
            if ml_allowed:
                reasons.append(f"No viable ML algorithm for '{analysis_type}' given this dataset — falling back to deterministic")
            candidates = self._filter_by_requirements(self._registry.deterministic_algorithms_for(analysis_type), dataset_context, target_columns)
            if preferred_algorithm:
                candidates = self._prioritize_preferred(candidates, preferred_algorithm, reasons)

        if not candidates:
            reasons.append(f"No registered algorithm (ML or deterministic) for '{analysis_type}' meets this dataset's requirements")
            return SelectedModel(algorithm=None, implementation_class=None, requires_ml=False, cost_tier=None, reasons=reasons)

        chosen = self._pick_within_expensive_budget(candidates, reasons)
        reasons.append(f"Selected '{chosen.algorithm}' (cost_tier={chosen.cost_tier}) from {len(candidates)} viable candidate(s)")
        if chosen.cost_tier == COST_EXPENSIVE:
            self._expensive_used += 1

        return SelectedModel(
            algorithm=chosen.algorithm,
            implementation_class=chosen.implementation_class,
            requires_ml=chosen.requirements.requires_ml,
            cost_tier=chosen.cost_tier,
            reasons=reasons,
        )

    @staticmethod
    def _prioritize_preferred(candidates: list[AlgorithmSpec], preferred_algorithm: str, reasons: list[str]) -> list[AlgorithmSpec]:
        preferred = [c for c in candidates if c.algorithm == preferred_algorithm]
        if not preferred:
            return candidates
        reasons.append(f"Domain plugin prefers '{preferred_algorithm}' for this analysis type — prioritized over the registry's default order")
        rest = [c for c in candidates if c.algorithm != preferred_algorithm]
        return preferred + rest

    def _filter_by_requirements(
        self, candidates: list[AlgorithmSpec], ctx: DatasetContext, target_columns: list[str],
    ) -> list[AlgorithmSpec]:
        has_temporal = any(c.is_temporal for c in ctx.columns)
        has_target = bool((ctx.feature_recommendation or {}).get("target_column"))
        survivors = []
        for spec in candidates:
            req = spec.requirements
            if ctx.row_count < req.min_rows:
                continue
            if req.requires_datetime and not has_temporal:
                continue
            if req.requires_target and not has_target:
                continue
            if len(target_columns) < req.min_feature_columns:
                continue
            survivors.append(spec)
        return survivors

    def _pick_within_expensive_budget(self, candidates: list[AlgorithmSpec], reasons: list[str]) -> AlgorithmSpec:
        budget_exhausted = self._expensive_used >= self._budget.max_expensive_operations
        if not budget_exhausted:
            return candidates[0]  # registry order = preference order

        cheaper = [c for c in candidates if c.cost_tier != COST_EXPENSIVE]
        if cheaper:
            reasons.append(
                f"Expensive-operation budget ({self._budget.max_expensive_operations}) exhausted — "
                f"chose a non-expensive alternative instead of the top-preference algorithm"
            )
            return cheaper[0]
        # nothing cheaper is available for this analysis type — proceed with
        # the preferred (expensive) one rather than produce no evidence at all.
        reasons.append(
            "Expensive-operation budget exhausted, but no non-expensive alternative exists for this analysis type — proceeding anyway"
        )
        return candidates[0]
