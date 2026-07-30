"""app/services/domain_plugins/base.py — Domain Enhancement Layer (plugin
architecture) for the Agent 3 redesign (plan "zany-giggling-crayon").

A DomainPlugin never replaces the generic engine — it only ever adds on
top of it ("Plugin → Enhance generic plan", not `if domain == X: ... else:
...`). When Agent 1 classifies a dataset into a domain with a matching
plugin, that plugin's curated content (KPIs, driver columns, preferred
deterministic strategies) takes priority; every other domain — including
ones with no plugin at all — still gets the full generic engine.

Two groups of methods:
  - The ones from the plan itself (applies_to, enhance_plan,
    get_kpi_definitions, get_intent_vocabulary, get_driver_columns,
    get_preferred_deterministic_strategy) — consumed starting in later
    phases once the new graph topology (Planner/Scheduler/Analyzers)
    exists.
  - Phase 2 additions (get_rule_engine_paths, get_driver_labels,
    get_view_name, get_variance_columns, get_ratio_columns) — consumed
    right now by nodes/pipeline.py's construction of RuleEngine/
    RootCauseTool/SQLTool, which is how the Insurance plugin proves it's
    genuinely driving today's behavior rather than being unused
    scaffolding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.services.question_interpreter.models import QuestionIntent


class DomainPlugin(ABC):
    @abstractmethod
    def applies_to(self, detected_domain: str) -> bool:
        """True if this plugin should activate for the given (already
        canonicalized) domain string, e.g. "Insurance"."""
        raise NotImplementedError

    def enhance_plan(self, plan: list, dataset_context: Any, question_intent: QuestionIntent | None = None) -> list:
        """Additive only — return `plan` with domain-specific analyses
        appended, never with generic ones removed. Default: no
        enhancement.

        question_intent (Phase 4) is what lets a plugin inject curated-KPI
        analyses (kpi_summary/kpi_variance/KPI-grounded root_cause) using
        question_intent.resolved_kpi_name — the generic AnalyticsPlanner
        never proposes these itself, since a curated KPI catalog is
        inherently domain-plugin content, not a structural dataset
        pattern."""
        return plan

    @abstractmethod
    def get_kpi_definitions(self) -> dict:
        """This domain's curated KPI set — {kpi_key: {label, actual_column,
        budget_column, ..., higher_is_better, category, ...}}, the same
        shape as today's config/rules/kpi_definitions.json."""
        raise NotImplementedError

    def get_intent_vocabulary(self) -> dict:
        """Optional additive intent keywords/aliases beyond the generic
        engine's own (e.g. Insurance's KPI aliases, dimension keywords).
        Default: none."""
        return {}

    def get_driver_columns(self) -> list[str] | None:
        """Pre-computed, pre-labeled variance-driver column names, for
        RootCauseTool's labeled-driver mode. None means this domain has no
        such columns — the generic correlation-based mode applies instead
        (a later phase)."""
        return None

    def get_driver_labels(self) -> dict[str, str] | None:
        """Display labels for get_driver_columns()'s columns. None means
        use the raw column names as labels."""
        return None

    def get_preferred_deterministic_strategy(self, analysis_type: str) -> str | None:
        """Pin this domain to a specific deterministic algorithm name
        (must match an AlgorithmSpec.algorithm in the model registry) for
        analysis_type, overriding the generic registry's default priority
        order. None means no preference — use the registry's default.
        This is what keeps a plugin's existing narrative/output byte-
        identical even as the generic engine's own defaults evolve (see
        the plan's "Improve Current Fallbacks" section)."""
        return None

    # ── Phase 2 additions — consumed today by nodes/pipeline.py ────────────

    def get_rule_engine_paths(self) -> dict[str, str]:
        """{'kpi_definitions_path', 'hierarchy_path', 'business_rules_path'}
        (any subset) to pass into RuleEngine's constructor. Empty dict
        means "use RuleEngine's own defaults" (today's exact Insurance
        paths, sourced from app.config)."""
        return {}

    def get_view_name(self) -> str:
        """DuckDB view name for SQLTool. Defaults to "insurance" purely
        for backward compatibility with existing SQL text/logs — a new
        plugin can use any name."""
        return "insurance"

    # ── Phase 4 addition ─────────────────────────────────────────────────

    def get_default_kpi_name(self) -> str | None:
        """The KPI to fall back to when a business_question mentions no
        specific KPI (by alias) and there's no prior-turn KPI to carry
        forward — the named home for what used to be nodes/pipeline.py's
        hardcoded `"underwriting_result"` literal. None means this domain
        has no sensible default (no curated-KPI query flow applies)."""
        return None
