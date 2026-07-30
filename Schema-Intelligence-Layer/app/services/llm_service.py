"""
LLM service using Azure OpenAI.
Handles dataset classification and column description generation.
Only metadata (column names, types, sample values) is sent to the LLM — never the full dataset.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import settings
from app.models.schemas import DatasetMetadata, LLMClassification
from app.prompts.llm_service_prompt import (
    get_column_descriptions_prompt,
    get_classify_dataset_prompt,
)

logger = logging.getLogger(__name__)

# Removed static VALID_DOMAINS and SUB_DOMAINS taxonomies.
# Domain and sub-domain classification now runs fully dynamically via the LLM prompts.

# A 141-column dataset needed ~6,283 tokens for one description prompt and
# ~7,597 for one classification prompt. 30 columns per LLM call keeps either
# prompt comfortably bounded regardless of provider-side token limits.
MAX_COLUMNS_PER_LLM_CALL = 30

# Description batches are independent LLM calls (disjoint column slices) --
# run them concurrently. Capped, not unbounded, for the same reason as the
# per-call token cap above: a safe multiple of one batch's request size
# rather than firing every batch of a very wide dataset at once.
_MAX_PARALLEL_BATCHES = 5



def _get_llm_client() -> httpx.Client:
    """Create and return a reusable HTTP client for Azure OpenAI chat completions."""
    return httpx.Client(timeout=30.0)


def _chat_complete(client: httpx.Client, messages: list[dict], temperature: float, max_tokens: int) -> str:
    response = client.post(
        f"{settings.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/chat/completions",
        headers={
            "api-key": settings.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": settings.AZURE_OPENAI_DEPLOYMENT,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _strip_json_fences(response_text: str) -> str:
    """Strip a markdown code fence (```json ... ``` or ``` ... ```) that the LLM
    sometimes wraps its JSON response in, despite being asked for JSON only."""
    if response_text.startswith("```"):
        response_text = response_text.strip("`").strip()
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()
    return response_text


def _build_metadata_context(metadata: DatasetMetadata) -> str:
    """
    Build a text summary of dataset metadata for the LLM prompt.
    """
    lines = [
        f"Dataset Name: {metadata.dataset_name}",
        f"File Type: {metadata.file_type}",
        f"Number of Rows: {metadata.row_count}",
        f"Number of Columns: {metadata.column_count}",
        "",
        "Columns:",
    ]

    # Pre-extract up to 3 distinct non-empty sample values per column from sample_data list
    col_samples = {col: [] for col in metadata.column_names}
    if metadata.sample_data:
        for row in metadata.sample_data:
            for col in metadata.column_names:
                val = row.get(col)
                if val is not None and val != "":
                    val_str = str(val)[:50]  # Limit length of a single sample value
                    if val_str not in col_samples[col] and len(col_samples[col]) < 3:
                        col_samples[col].append(val_str)

    for i, col_name in enumerate(metadata.column_names):
        dtype = metadata.column_data_types[i] if i < len(metadata.column_data_types) else "unknown"
        samples = col_samples.get(col_name, [])
        samples_str = f", samples: {samples}" if samples else ""
        lines.append(f"  - {col_name} (type: {dtype}{samples_str})")

    return "\n".join(lines)


def _generate_column_descriptions_batch(
    client: httpx.Client, batch_metadata: DatasetMetadata, batch_names: list[str], batch_types: list[str]
) -> dict[str, str]:
    """Generate descriptions for a single batch of columns. Falls back to placeholder
    text for just this batch on any failure, so one bad batch can't wipe out others."""
    context = _build_metadata_context(batch_metadata)
    prompt = get_column_descriptions_prompt(context)

    try:
        content = _chat_complete(
            client,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis assistant. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        response_text = _strip_json_fences(content.strip())
        descriptions = json.loads(response_text)

        return {col: descriptions.get(col, f"Column '{col}' in the dataset.") for col in batch_names}

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM column descriptions as JSON: {e}")
        return {col: f"Column '{col}' of type {batch_types[i]}" for i, col in enumerate(batch_names)}
    except Exception as e:
        logger.error(f"Azure OpenAI API error during column description generation: {e}")
        return {col: f"Column '{col}' of type {batch_types[i]}" for i, col in enumerate(batch_names)}


def generate_column_descriptions(metadata: DatasetMetadata) -> dict[str, str]:
    """
    Generate short descriptions for each column using Azure OpenAI.

    Wide datasets are split into batches of MAX_COLUMNS_PER_LLM_CALL columns so a single
    prompt never exceeds the provider's per-request token limit — each batch is its own LLM call.
    Batches are independent (disjoint columns), so they run concurrently and are merged
    back into one dict covering every column.
    """
    client = _get_llm_client()
    all_descriptions: dict[str, str] = {}

    batch_bounds = list(range(0, len(metadata.column_names), MAX_COLUMNS_PER_LLM_CALL))
    batches = []
    for start in batch_bounds:
        batch_names = metadata.column_names[start:start + MAX_COLUMNS_PER_LLM_CALL]
        batch_types = metadata.column_data_types[start:start + MAX_COLUMNS_PER_LLM_CALL]
        batch_metadata = metadata.model_copy(update={
            "column_names": batch_names,
            "column_data_types": batch_types,
            "column_count": len(batch_names),
        })
        batches.append((batch_metadata, batch_names, batch_types))

    with ThreadPoolExecutor(max_workers=min(len(batches), _MAX_PARALLEL_BATCHES)) as executor:
        futures = [
            executor.submit(_generate_column_descriptions_batch, client, batch_metadata, batch_names, batch_types)
            for batch_metadata, batch_names, batch_types in batches
        ]
        for future in futures:
            all_descriptions.update(future.result())

    return all_descriptions


def classify_dataset(metadata: DatasetMetadata) -> LLMClassification:
    """
    Classify the dataset into a business domain using Azure OpenAI.

    Unlike column descriptions, classification produces one result for the whole
    dataset, so it can't be batched-and-merged. Wide datasets instead use only the
    first MAX_COLUMNS_PER_LLM_CALL columns as a representative sample — enough
    signal to classify the domain without needing exhaustive column coverage —
    keeping the prompt under the provider's per-request token limit.
    """
    client = _get_llm_client()

    capped_names = metadata.column_names[:MAX_COLUMNS_PER_LLM_CALL]
    capped_types = metadata.column_data_types[:MAX_COLUMNS_PER_LLM_CALL]
    capped_metadata = metadata.model_copy(update={
        "column_names": capped_names,
        "column_data_types": capped_types,
        "column_count": len(capped_names),
    })

    context = _build_metadata_context(capped_metadata)

    # Include column descriptions if available (capped to the same column subset)
    desc_section = ""
    if metadata.column_descriptions:
        desc_lines = ["", "Column Descriptions:"]
        for col in capped_names:
            desc = metadata.column_descriptions.get(col)
            if desc:
                desc_lines.append(f"  - {col}: {desc}")
        desc_section = "\n".join(desc_lines)

    prompt = get_classify_dataset_prompt(context, desc_section)

    try:
        content = _chat_complete(
            client,
            messages=[
                {
                    "role": "system",
                    "content": "You are a business data classification expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_tokens=500,
        )

        response_text = _strip_json_fences(content.strip())
        result = json.loads(response_text)

        # Retrieve classification from LLM response (fallback to defaults if empty/missing)
        domain = result.get("business_domain", "Other")
        if not isinstance(domain, str) or not domain.strip():
            domain = "Other"

        sub_domain = result.get("sub_domain", "General")
        if not isinstance(sub_domain, str) or not sub_domain.strip():
            sub_domain = "General"

        return LLMClassification(
            business_domain=domain.strip(),
            sub_domain=sub_domain.strip(),
            dataset_summary=result.get("dataset_summary", ""),
            confidence=result.get("confidence"),
            reason=result.get("reason"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM classification response as JSON: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM response parsing failed.",
            confidence=0.0,
            reason=str(e),
        )
    except Exception as e:
        logger.error(f"Azure OpenAI API error during dataset classification: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM service error.",
            confidence=0.0,
            reason=str(e),
        )

