"""Real LangChain tools for the dataset-classification agent. Built per-run via
a factory function since each run's metadata differs — there's no shared global
state a plain module-level @tool could safely close over.

Each tool wraps the existing, already-tested Azure OpenAI-calling implementation in
app/services/llm_service.py (batching for wide datasets, per-task temperature/
max_tokens, graceful per-batch fallback) rather than reimplementing it — the
only thing moving to LangChain here is *whether/when* these calls happen
(a real tool-call decision now, not a hardcoded function call in the route).
"""

import json
from typing import Any

from app.models.schemas import DatasetMetadata
from app.services.llm_service import classify_dataset as _classify_dataset_impl
from app.services.llm_service import generate_column_descriptions as _generate_descriptions_impl
from langchain_core.tools import tool


def build_classification_tools(metadata: DatasetMetadata) -> list:
    """
    Tools available to the classification agent. classify_dataset benefits from
    column descriptions as context, so it reads whatever generate_column_descriptions
    already produced this run (via the shared holder below) — same behavior as the
    original code's "include descriptions if available" branch.
    """
    holder: dict[str, Any] = {"descriptions": None}

    @tool
    def generate_column_descriptions() -> str:
        """Generate a short, one-sentence description for every column in this
        dataset via the LLM, grounded in each column's name, type, and sample
        values. Returns a JSON object mapping column name to description. Call
        this before classify_dataset — descriptions improve classification
        accuracy."""
        descriptions = _generate_descriptions_impl(metadata)
        holder["descriptions"] = descriptions
        return json.dumps(descriptions)

    @tool
    def classify_dataset() -> str:
        """Classify this dataset into a business domain and sub-domain via the
        LLM, using column names/types and (if already generated) column
        descriptions as evidence. Returns a JSON object with business_domain,
        sub_domain, dataset_summary, confidence, and reason."""
        metadata_with_descriptions = metadata
        if holder["descriptions"]:
            metadata_with_descriptions = metadata.model_copy(
                update={"column_descriptions": holder["descriptions"]}
            )
        result = _classify_dataset_impl(metadata_with_descriptions)
        return json.dumps(result.model_dump())

    return [generate_column_descriptions, classify_dataset]


def tools_to_registry(tools: list) -> dict[str, Any]:
    """Build a name->tool lookup so agent.yaml's agent_tools list (names only)
    can be resolved into actual callables at graph-build time."""
    return {t.name: t for t in tools}
