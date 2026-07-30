"""tests/test_explanation_tool.py — Tests for app/services/explanation_tool.py's
deterministic confidence handling.

Confidence is a rule-based function of which evidence keys are present, not
something that should ever be left to the LLM's judgment (the tool's own
docstring: "Must ONLY narrate the evidence passed to it"). These tests cover
_compute_confidence directly, and _enforce_confidence_section's guarantee
that whatever an LLM writes in a "## Confidence" section gets overridden
with the deterministic value — added after Groq was observed self-labeling
a query "HIGH" confidence when only actuals + variance were present (no
root-cause evidence), contradicting agent.yaml's own confidence_rules.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tools.explanation_tool import ExplanationTool


# ── _compute_confidence ───────────────────────────────────────────────────────

def test_confidence_high_requires_actuals_variance_and_root_cause():
    level, text = ExplanationTool._compute_confidence({
        "actual": 1.0, "variance_amount": 1.0, "primary_driver": {"label": "x"},
    })
    assert level == "HIGH"
    assert "HIGH" in text


def test_confidence_medium_without_root_cause():
    level, text = ExplanationTool._compute_confidence({
        "actual": 1.0, "variance_amount": 1.0,
    })
    assert level == "MEDIUM"
    assert "root cause not analysed" in text


def test_confidence_medium_actuals_only():
    level, text = ExplanationTool._compute_confidence({"actual": 1.0})
    assert level == "MEDIUM"
    assert "comparison data absent" in text


def test_confidence_low_with_no_relevant_evidence():
    level, text = ExplanationTool._compute_confidence({"kpi": "Loss Ratio"})
    assert level == "LOW"


# ── _enforce_confidence_section ───────────────────────────────────────────────

def test_enforce_replaces_wrong_llm_confidence():
    # Reproduces the exact bug: Groq self-labeled HIGH for evidence that
    # only has actuals + variance (no root-cause), contradicting the
    # deterministic rule.
    groq_output = (
        "## Summary\nGWP is up 4.48%.\n\n"
        "## Confidence\nHIGH, as the evidence provides a clear and complete picture."
    )
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0, "variance_amount": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert "MEDIUM" in fixed
    assert "clear and complete picture" not in fixed
    assert "## Summary\nGWP is up 4.48%." in fixed  # rest of the narrative untouched


def test_enforce_appends_section_if_llm_omitted_it():
    groq_output = "## Summary\nGWP is up 4.48%."
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert fixed.startswith(groq_output)
    assert "## Confidence" in fixed
    assert correct_text in fixed


def test_enforce_is_case_insensitive_on_heading():
    groq_output = "## summary\ntext\n\n## confidence\nwrong text here"
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert "wrong text here" not in fixed
    assert correct_text in fixed


# ── _template_format — multi-analysis "report mode" (Phase 4) ──────────────

def test_template_format_renders_each_nested_analysis_under_its_own_heading():
    """EvidenceBuilder.to_narration_context()'s multi-analysis shape nests
    each analysis's own flat evidence under analyses[analysis_type] — the
    template formatter must not render this as an empty report."""
    tool = ExplanationTool(llm_readiness_score=0.0)  # force template path, no LLM client needed
    evidence = {
        "analyses": {
            "trend": {"direction": "increasing", "overall_change_pct": 12.5},
            "segmentation": {"segments": {"Low": 3, "High": 7}, "total_records": 10},
        }
    }
    response = tool.narrate(evidence, "Analyze this dataset")

    assert "# Trend" in response
    assert "# Segmentation" in response
    assert "Direction" in response and "increasing" in response
    assert "Risk Segmentation Profiles" in response  # segments dict still gets its special rendering
    assert "Low" in response and "High" in response
    # Only the combined report's own top-level context line survives —
    # not one repeated per nested section.
    assert response.count("Query context:") == 1


def test_template_format_empty_analyses_dict_falls_through_to_flat_path():
    tool = ExplanationTool(llm_readiness_score=0.0)
    response = tool.narrate({"analyses": {}}, "q")
    assert "Analysis completed for query" in response


# ── _template_format — group comparison rendering (Phase 4 follow-up) ──────

def test_template_format_renders_group_comparison_not_just_scalar_fields():
    """Regression test: ComparativeAnalyzer's evidence (groups/highest/
    lowest — AnalyticsTool.compare_groups()) used to render as just
    'Group Count'/'Method' with the actual comparison silently dropped,
    since groups/highest/lowest are lists/dicts the generic scalar-only
    key-value loop skips. Caught via live testing against a Banking
    dataset with no domain plugin."""
    tool = ExplanationTool(llm_readiness_score=0.0)
    evidence = {
        "groups": [
            {"region": "EMEA", "revenue": 500.0, "rank": 1},
            {"region": "APAC", "revenue": 300.0, "rank": 2},
        ],
        "highest": {"region": "EMEA", "revenue": 500.0, "rank": 1},
        "lowest": {"region": "APAC", "revenue": 300.0, "rank": 2},
        "group_count": 2,
        "method": "sum",
    }
    response = tool.narrate(evidence, "Compare revenue by region")

    assert "## Group Comparison" in response
    assert "EMEA" in response and "500.0" in response
    assert "APAC" in response and "300.0" in response
    assert "Highest" in response
    assert "Lowest" in response
    # Still renders the scalar fields too — additive, not a replacement.
    assert "Group Count" in response


# ── Azure OpenAI path / template formatter fallback ─────────────────────────

def test_azure_openai_used_when_client_configured_and_readiness_passes():
    tool = ExplanationTool(llm_readiness_score=99.0)
    tool._client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "## Summary\nAzure OpenAI said hi.\n## Confidence\nignored"}}]
    }
    tool._client.post.return_value = mock_response
    response = tool.narrate({"variance_amount": 5}, "Why?")

    assert tool.last_engine_used == "Azure OpenAI"
    assert "Azure OpenAI said hi." in response
    tool._client.post.assert_called_once()


def test_falls_back_to_template_formatter_when_azure_openai_call_fails():
    tool = ExplanationTool(llm_readiness_score=99.0)
    tool._client = MagicMock()
    tool._client.post.side_effect = Exception("azure openai down")

    response = tool.narrate({"variance_amount": 5}, "Why?")

    assert tool.last_engine_used == "Template Formatter (Azure OpenAI error)"
    assert response  # template formatter still produces something


def test_no_azure_openai_client_goes_straight_to_template_formatter():
    tool = ExplanationTool(llm_readiness_score=99.0)
    tool._client = None  # simulate Azure OpenAI credentials not configured

    response = tool.narrate({"variance_amount": 5}, "Why?")

    assert tool.last_engine_used == "Template Formatter"
    assert response
