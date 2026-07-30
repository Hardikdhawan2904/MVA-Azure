"""app/services/question_interpreter/models.py — Stage 3 of the Agent 3
redesign (plan "zany-giggling-crayon"): the QuestionIntent shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionIntent:
    candidate_analysis_types: set[str]   # narrows the Planner's candidates; never empty, never adds impossible types
    mentioned_kpis: list[str] = field(default_factory=list)
    rationale: str = ""
    # Phase 4 (plan "zany-giggling-crayon") additions — populated by the
    # interpret_question graph node *after* calling
    # BusinessQuestionInterpreter.interpret() (this class stays domain-
    # agnostic and untouched; KPI-name/filter resolution is domain-plugin-
    # specific, see DomainPlugin.get_intent_vocabulary()). resolved_kpi_name
    # is the curated KPI (by RuleEngine key/alias) this question is about,
    # if any — distinct from mentioned_kpis, which is Stage 2's generic
    # discovered-KPI substring match. filters carries period/dimension
    # filters (fiscal_year, region, ...) extracted the same way
    # nodes/pipeline.py's old _extract_filters did.
    resolved_kpi_name: str | None = None
    filters: dict = field(default_factory=dict)
    # Phase 4.6 additions — populated by BusinessQuestionInterpreter from a
    # direct phrase match against DatasetContext's metric-role/
    # dimension-role column names (not just discovered-KPI names, which
    # mentioned_kpis already covers). Best-match first. Stage 4's planning
    # rules prefer these over their own "first structurally valid column"
    # default when present, instead of always picking whichever column
    # happens to appear first in the dataset regardless of what the
    # question actually asked about. Named for *what* they are (a metric/
    # dimension the question prefers), not *how* they were derived — a
    # later phase could populate them via KPI aliases, plugin metadata, or
    # an LLM instead of a keyword/phrase match without Stage 4 changing at
    # all.
    preferred_metrics: list[str] = field(default_factory=list)
    preferred_dimensions: list[str] = field(default_factory=list)
