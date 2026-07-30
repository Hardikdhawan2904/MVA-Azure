"""Pydantic models for structured LLM outputs."""

from pydantic import BaseModel, Field


class ColumnSemanticDecision(BaseModel):
    """LLM decision for a single column's semantic type and role."""
    column_name: str
    decision: str = Field(description="One of: confirmed, overridden, unresolved")
    confirmed_semantic_type: str | None = None
    confirmed_column_role: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    recommended_mandatory: bool | None = None
    recommended_expected_unique: bool | None = None


class SchemaIntelligenceBatchResponse(BaseModel):
    """LLM batch response for multiple columns."""
    columns: list[ColumnSemanticDecision]
    model_name: str = ""
    prompt_version: str = "si-v1"


class SecondaryDomainDecision(BaseModel):
    """LLM decision for secondary domain classification."""
    selected_domain: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    evidence: list[str] = []


class RuleSuggestionOutput(BaseModel):
    """
    A single LLM-proposed business rule.

    Fields mirror the raw dict shape RuleLoader._extract_parameters() reads for
    each RuleType (the same shape used by YAML business_rules in config/domains/*.yaml),
    so a suggestion can be persisted and, once approved, loaded straight back into
    the rule engine with no translation layer. Only the fields relevant to `type`
    need to be populated by the LLM — the rest stay at their defaults.
    """
    rule_key: str
    type: str = Field(description="One of the supported rule types listed in the prompt")
    description: str
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str = "medium"
    target_columns: list[str] = Field(default_factory=list, description="Columns this rule references, for display")

    # non_null / expected_unique / regex_match / allowed_values / numeric_range
    target_column: str | None = None
    pattern: str | None = None
    values: list[str] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    inclusive_min: bool = True
    inclusive_max: bool = True

    # column_comparison
    left_column: str | None = None
    operator: str | None = None
    right_column: str | None = None

    # conditional_required
    condition_column: str | None = None
    condition_value: str | None = None
    required_column: str | None = None


class RuleSuggestionBatch(BaseModel):
    """Batch of LLM-proposed rules."""
    suggestions: list[RuleSuggestionOutput]
