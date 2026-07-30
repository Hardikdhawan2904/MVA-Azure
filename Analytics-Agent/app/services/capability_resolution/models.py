"""app/services/capability_resolution/models.py — Stage 1 of the Agent 3
redesign (plan "zany-giggling-crayon"): the CapabilityProfile shape.

Every capability's result is split into two independently-reasoned parts:
StructuralResult (can this analysis mechanically run at all, given the
dataset's shape — Agent 3's own concern, zero overlap with Agent 2's
readiness scoring) and ExecutionResult (does Agent 2's own ml_readiness
score clear Agent 3's own existing execution threshold — the ONLY place
Agent 2's score is consulted, never recomputed). See
AnalyticsCapabilityResolver's module docstring for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The capability names this stage resolves — plain strings, not an enum, so
# downstream stages (Planner, Selector) can key off them the same way
# AnalysisType strings are used throughout the rest of this redesign.
#
# Note: anomaly_detection is included here even though the plan's own
# "8 capabilities" summary line dropped it — an inconsistency introduced
# across the plan's editing rounds. It's clearly intended throughout the
# rest of the document (the Planner's rule table plans it, "Improve
# Current Fallbacks" dedicates a whole subsection to its deterministic
# strategies, and it was one of the illustrative examples in the original
# request). Corrected here rather than silently reproduced.
FORECASTING = "forecasting"
CLUSTERING = "clustering"
CLASSIFICATION = "classification"
REGRESSION = "regression"
SEGMENTATION = "segmentation"
CORRELATION = "correlation"
ASSOCIATION_RULES = "association_rules"
TIME_SERIES = "time_series"
ANOMALY_DETECTION = "anomaly_detection"

ALL_CAPABILITIES = (
    FORECASTING, CLUSTERING, CLASSIFICATION, REGRESSION,
    SEGMENTATION, CORRELATION, ASSOCIATION_RULES, TIME_SERIES,
    ANOMALY_DETECTION,
)


@dataclass
class StructuralResult:
    supported: bool
    reason: str  # always present — states what was found, whether pass or fail


@dataclass
class ExecutionResult:
    supported: bool
    confidence: float | None = None  # only when supported=True; == ml_readiness_score / 100.0, never recomputed
    reason: str | None = None        # only when supported=False


@dataclass
class CapabilityResult:
    structural: StructuralResult
    # None exactly when structural.supported is False — there is nothing
    # to gate execution against if the analysis is mechanically impossible.
    execution: ExecutionResult | None = None


# Maps the "analysis_type" vocabulary used by Stages 3-7 (Interpreter,
# Planner, Scheduler, Analyzers — e.g. "forecast", "time_series_analysis")
# onto this module's 9 capability names (e.g. "forecasting", "time_series").
# The two vocabularies differ slightly (analysis types match today's
# existing intent/analyzer names; capabilities are the resolver's own
# concept) — this is the single place they're reconciled, so every stage
# that needs to ask "is analysis_type X structurally/execution possible"
# imports this instead of re-deriving its own mapping.
#
# Analysis types with NO entry here (root_cause, feature_importance,
# comparative_analysis, ranking, distribution_analysis, trend) have no
# capability gate at all — they're always structurally available /
# deterministic-first by design (see Stages 5-7 of the plan).
ANALYSIS_TYPE_TO_CAPABILITY: dict[str, str] = {
    "forecast": FORECASTING,
    "time_series_analysis": TIME_SERIES,
    "clustering": CLUSTERING,
    "classification": CLASSIFICATION,
    "regression": REGRESSION,
    "segmentation": SEGMENTATION,
    "correlation": CORRELATION,
    "association_rules": ASSOCIATION_RULES,
    "anomaly_detection": ANOMALY_DETECTION,
}


def capability_for_analysis_type(analysis_type: str) -> str | None:
    """None means this analysis type has no capability gate — always
    structurally available (see ANALYSIS_TYPE_TO_CAPABILITY's docstring)."""
    return ANALYSIS_TYPE_TO_CAPABILITY.get(analysis_type)


@dataclass
class CapabilityProfile:
    capabilities: dict[str, CapabilityResult] = field(default_factory=dict)

    def is_structurally_possible(self, capability: str) -> bool:
        result = self.capabilities.get(capability)
        return bool(result and result.structural.supported)

    def is_ml_viable(self, capability: str) -> bool:
        result = self.capabilities.get(capability)
        return bool(result and result.execution and result.execution.supported)

    def confidence_for(self, capability: str) -> float | None:
        result = self.capabilities.get(capability)
        if result and result.execution and result.execution.supported:
            return result.execution.confidence
        return None

    def is_analysis_type_possible(self, analysis_type: str) -> bool:
        """What Stage 3 (Interpreter) and Stage 4 (Planner) actually call —
        translates an analysis_type (e.g. "forecast") via
        capability_for_analysis_type() and checks structural support; an
        analysis type with no capability gate (root_cause, trend, ...) is
        always possible from this stage's point of view."""
        capability = capability_for_analysis_type(analysis_type)
        if capability is None:
            return True
        return self.is_structurally_possible(capability)

    def is_analysis_type_ml_viable(self, analysis_type: str) -> bool:
        """What Stage 6 (ModelSelector) calls to decide ML vs. deterministic
        for an already-planned analysis. An analysis type with no capability
        gate (root_cause, trend, ...) is deterministic-first by design (see
        ANALYSIS_TYPE_TO_CAPABILITY's docstring) — never ML-viable via this
        mechanism, not because it's blocked, but because there's no
        readiness-gated model behind it in the first place."""
        capability = capability_for_analysis_type(analysis_type)
        if capability is None:
            return False
        return self.is_ml_viable(capability)
