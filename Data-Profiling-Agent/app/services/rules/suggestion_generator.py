"""Rule suggestion generator — proposes candidate rules via LLM."""

from typing import Any
import uuid

from app.core.enums import RuleSuggestionStatus, RuleType
from app.core.logging import get_logger
from app.services.llm.interface import LLMProvider, LLMRequest
from app.services.llm.prompts import RULE_SUGGESTION_SYSTEM, RULE_SUGGESTION_PROMPT_V1
from app.services.llm.structured_output import RuleSuggestionBatch
from app.services.profiling.column_profiler import ColumnProfileResult

logger = get_logger(__name__)

# Rule types RuleEngine actually evaluates today. date_range, cross_field_equality, and
# cross_field_inequality exist in RuleType but have no evaluator yet, so suggesting them
# would produce a rule that only ever errors with "Unsupported rule type" once approved.
ENGINE_SUPPORTED_RULE_TYPES = {
    RuleType.NON_NULL,
    RuleType.EXPECTED_UNIQUE,
    RuleType.REGEX_MATCH,
    RuleType.ALLOWED_VALUES,
    RuleType.NUMERIC_RANGE,
    RuleType.COLUMN_COMPARISON,
    RuleType.CONDITIONAL_REQUIRED,
}


class SuggestedRule:
    """
    An AI-proposed rule that requires approval before scoring.

    `definition` is the raw dict shape RuleLoader._extract_parameters() reads for the
    given `type` — the same shape used by YAML business_rules in config/domains/*.yaml.
    Storing it verbatim means an approved suggestion can be fed straight into
    RuleLoader.load_request_rules() later with no translation layer.

    `engine_compatible` is False when the LLM emitted a `type` the engine can't evaluate
    (invalid enum value, or a defined-but-unimplemented type) — the suggestion is still
    stored for visibility, but the API should surface this so it isn't approved blindly.
    """

    def __init__(
        self,
        suggestion_id: str,
        definition: dict[str, Any],
        confidence: float,
        engine_compatible: bool,
        status: RuleSuggestionStatus = RuleSuggestionStatus.PROPOSED,
    ):
        self.suggestion_id = suggestion_id
        self.definition = definition
        self.confidence = confidence
        self.engine_compatible = engine_compatible
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            **self.definition,
            "confidence": round(self.confidence, 4),
            "engine_compatible": self.engine_compatible,
            "status": self.status.value,
        }


class RuleSuggestionGenerator:
    """Generates rule suggestions via LLM. Suggestions are NEVER auto-activated."""

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    def generate(
        self,
        profiles: list[ColumnProfileResult],
        primary_domain: str,
        secondary_domain: str | None,
    ) -> list[SuggestedRule]:
        """
        Generate rule suggestions using the LLM.

        All returned suggestions have status=proposed.
        They do NOT participate in scoring until approved.
        """
        columns_summary = self._build_columns_summary(profiles)

        prompt = RULE_SUGGESTION_PROMPT_V1.format(
            primary_domain=primary_domain,
            secondary_domain=secondary_domain or "unknown",
            columns_summary=columns_summary,
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=RULE_SUGGESTION_SYSTEM,
            temperature=0.2,
            max_tokens=2000,
        )

        parsed, response = self._llm.complete_structured(request, RuleSuggestionBatch)
        if parsed is None:
            logger.warning("rule_suggestion_llm_failed", error=response.error)
            return []

        suggestions: list[SuggestedRule] = []
        for s in parsed.suggestions[:5]:  # Max 5 suggestions
            try:
                engine_compatible = RuleType(s.type) in ENGINE_SUPPORTED_RULE_TYPES
            except ValueError:
                engine_compatible = False
                logger.warning("rule_suggestion_invalid_type", rule_key=s.rule_key, type=s.type)

            suggestions.append(SuggestedRule(
                suggestion_id=str(uuid.uuid4()),
                definition=s.model_dump(),
                confidence=s.confidence,
                engine_compatible=engine_compatible,
                status=RuleSuggestionStatus.PROPOSED,
            ))

        return suggestions

    def _build_columns_summary(self, profiles: list[ColumnProfileResult]) -> str:
        """Build bounded column summary for the LLM prompt."""
        lines: list[str] = []
        for p in profiles[:30]:  # Bounded
            lines.append(
                f"- {p.physical_name}: type={p.pandas_dtype}, "
                f"nulls={p.null_ratio:.2%}, "
                f"distinct={p.distinct_count}, "
                f"samples={p.representative_values[:3]}"
            )
        return "\n".join(lines)
