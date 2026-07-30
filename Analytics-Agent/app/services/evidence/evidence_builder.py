"""app/services/evidence/evidence_builder.py — Stage 8 of the Agent 3
redesign (plan "zany-giggling-crayon"): Evidence / EvidenceBuilder.

Every analyzer (Stage 7) returns one Evidence object instead of a bare
dict — evidence, confidence, metrics, charts, model_metadata,
fallback_metadata, and reasons are independently addressable instead of
being flattened together the way today's handlers do it. Backward-compat
mapping (see the plan's Stage 8 section): today's flat evidence dict
becomes Evidence.fallback_metadata verbatim on the deterministic path,
and today's numeric payload becomes Evidence.evidence verbatim — so
ExplanationTool (Stage 9, not redesigned) needs at most a one-line
adapter via to_narration_context(), not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    charts: list[dict[str, Any]] | None = None
    model_metadata: dict[str, Any] | None = None
    fallback_metadata: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)

    def flatten(self) -> dict[str, Any]:
        """The single-analysis backward-compat shape: fallback_metadata's
        keys (ml_readiness_blocked/fallback_reason/fallback_applied — mirrors
        today's flat evidence dict) merged with evidence's numeric payload,
        evidence taking precedence on any key collision. This is exactly
        what ExplanationTool.narrate() already knows how to read."""
        flat: dict[str, Any] = {}
        if self.fallback_metadata:
            flat.update(self.fallback_metadata)
        flat.update(self.evidence)
        return flat


class EvidenceBuilder:
    """Accumulates one Evidence per scheduled (not merely planned) analysis
    — budget-bounded by the Scheduler (Stage 5), trivial memory overhead."""

    def __init__(self):
        self._entries: list[tuple[str, str, Evidence]] = []

    def add(self, analysis_type: str, analyzer_name: str, evidence: Evidence) -> None:
        self._entries.append((analysis_type, analyzer_name, evidence))

    def all(self) -> list[tuple[str, str, Evidence]]:
        return list(self._entries)

    def by_type(self, analysis_type: str) -> list[Evidence]:
        return [e for (t, _n, e) in self._entries if t == analysis_type]

    def is_empty(self) -> bool:
        return not self._entries

    def to_narration_context(self) -> dict[str, Any]:
        """The ONLY payload ExplanationTool may read (Stage 9's contract).

        A single scheduled analysis (today's one-question-one-answer flow,
        and the common case even under the new pipeline) flattens to
        exactly the shape ExplanationTool.narrate() already expects — no
        adapter needed at the call site.

        Multiple scheduled analyses (report mode — a business_question-less
        request against a wide dataset) nest each analysis's flattened
        evidence under its analysis_type, so no analysis's keys can
        collide with another's. How ExplanationTool eventually narrates
        multi-analysis evidence is a Phase 4 wiring decision; this method
        only guarantees the shape is well-defined and lossless.
        """
        if not self._entries:
            return {}
        if len(self._entries) == 1:
            return self._entries[0][2].flatten()
        return {
            "analyses": {
                analysis_type: analyzer_evidence.flatten()
                for analysis_type, _analyzer_name, analyzer_evidence in self._entries
            }
        }
