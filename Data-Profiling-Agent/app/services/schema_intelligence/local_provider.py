"""Local Schema Intelligence provider — uses LLM for semantic reasoning."""

import json
from concurrent.futures import ThreadPoolExecutor

from app.core.logging import get_logger
from app.core.enums import SchemaIntelligenceDecision
from app.services.llm.interface import LLMProvider, LLMRequest
from app.services.llm.prompts import SCHEMA_INTELLIGENCE_SYSTEM, SCHEMA_INTELLIGENCE_PROMPT_V1
from app.services.llm.structured_output import SchemaIntelligenceBatchResponse
from app.services.schema_intelligence.models import (
    ColumnAnalysisInput,
    ColumnAnalysisResult,
    DomainContext,
    SchemaIntelligenceResult,
)

logger = get_logger(__name__)

# Maximum columns per LLM batch to avoid token limits
_MAX_BATCH_SIZE = 30

# Batches are independent LLM calls (different columns, same domain_context) --
# run them concurrently instead of one-at-a-time. Capped, not unbounded, to
# stay a safe multiple of what a single-batch burst already sends to the LLM
# rather than firing every batch of a very wide dataset at once.
_MAX_PARALLEL_BATCHES = 5


class LocalSchemaIntelligenceProvider:
    """
    Local implementation of Schema Intelligence using an LLM.

    Falls back to deterministic candidates on LLM failure.
    """

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    def analyze(
        self,
        columns: list[ColumnAnalysisInput],
        domain_context: DomainContext,
    ) -> SchemaIntelligenceResult:
        """
        Analyze columns using the LLM for semantic confirmation/override.

        On LLM failure, returns a fallback result accepting all candidates as-is.
        """
        if not columns:
            return SchemaIntelligenceResult(
                column_results=[],
                success=True,
                fallback_used=False,
            )

        # Process in batches. Each batch is an independent LLM call (a
        # disjoint slice of columns against the same domain_context), so
        # they run concurrently — a 141-column dataset's 5 batches complete
        # in roughly one batch's latency instead of five.
        all_results: list[ColumnAnalysisResult] = []
        model_name = ""
        prompt_version = "si-v1"

        batches = [
            columns[start:start + _MAX_BATCH_SIZE]
            for start in range(0, len(columns), _MAX_BATCH_SIZE)
        ]

        with ThreadPoolExecutor(max_workers=min(len(batches), _MAX_PARALLEL_BATCHES)) as executor:
            futures = [executor.submit(self._analyze_batch, batch, domain_context) for batch in batches]
            batch_outcomes = [future.result() for future in futures]  # blocks here, in submission order

        for batch_start, batch, (batch_results, batch_model) in zip(
            range(0, len(columns), _MAX_BATCH_SIZE), batches, batch_outcomes
        ):
            if batch_results is None:
                # LLM failed — use deterministic fallback for entire batch
                logger.warning(
                    "schema_intelligence_fallback",
                    batch_start=batch_start,
                    batch_size=len(batch),
                )
                fallback = self._deterministic_fallback(batch)
                all_results.extend(fallback)
            else:
                all_results.extend(batch_results)
                model_name = batch_model or model_name

        # Check if any fallback was used
        fallback_used = any(
            "fallback" in (r.reasoning or "")
            for r in all_results
        )

        return SchemaIntelligenceResult(
            column_results=all_results,
            model_name=model_name,
            prompt_version=prompt_version,
            success=True,
            fallback_used=fallback_used,
        )

    def _analyze_batch(
        self,
        columns: list[ColumnAnalysisInput],
        domain_context: DomainContext,
    ) -> tuple[list[ColumnAnalysisResult] | None, str]:
        """Analyze a batch of columns via LLM. Returns None on failure."""
        columns_data = []
        for col in columns:
            columns_data.append({
                "column_name": col.column_name,
                "normalized_key": col.normalized_key,
                "description": col.description,
                "refined_physical_type": col.refined_physical_type,
                "statistics": {
                    k: v for k, v in col.statistics_summary.items()
                    if k in (
                        "null_ratio", "cardinality_ratio", "distinct_count",
                        "numeric_parse_ratio", "datetime_parse_ratio",
                        "min_value", "max_value", "dominant_patterns",
                    )
                },
                "sample_values": col.representative_sample_values[:5],
                "candidate_semantic_type": col.candidate_semantic_type,
                "candidate_column_role": col.candidate_column_role,
                "candidate_confidence": col.candidate_confidence,
            })

        prompt = SCHEMA_INTELLIGENCE_PROMPT_V1.format(
            primary_domain=domain_context.primary_domain,
            row_count=domain_context.row_count,
            column_count=domain_context.column_count,
            columns_json=json.dumps(columns_data, indent=2, default=str),
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=SCHEMA_INTELLIGENCE_SYSTEM,
            temperature=0.1,
            max_tokens=3000,
        )

        parsed, response = self._llm.complete_structured(
            request, SchemaIntelligenceBatchResponse
        )

        if parsed is None:
            logger.warning("schema_intelligence_llm_failed", error=response.error)
            return None, ""

        # Map LLM response to results
        results: list[ColumnAnalysisResult] = []
        col_map = {col.column_name: col for col in columns}

        for decision in parsed.columns:
            col_input = col_map.get(decision.column_name)
            if not col_input:
                continue

            # Map decision string to enum
            try:
                decision_enum = SchemaIntelligenceDecision(decision.decision)
            except ValueError:
                decision_enum = SchemaIntelligenceDecision.UNRESOLVED

            # If confirmed, use candidate values; if overridden, use new values
            if decision_enum == SchemaIntelligenceDecision.CONFIRMED:
                sem_type = col_input.candidate_semantic_type
                col_role = col_input.candidate_column_role
            elif decision_enum == SchemaIntelligenceDecision.OVERRIDDEN:
                sem_type = decision.confirmed_semantic_type or col_input.candidate_semantic_type
                col_role = decision.confirmed_column_role or col_input.candidate_column_role
            else:
                sem_type = col_input.candidate_semantic_type
                col_role = col_input.candidate_column_role

            results.append(ColumnAnalysisResult(
                column_name=decision.column_name,
                decision=decision_enum,
                confirmed_semantic_type=sem_type,
                confirmed_column_role=col_role,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                recommended_mandatory=decision.recommended_mandatory,
                recommended_expected_unique=decision.recommended_expected_unique,
            ))

        # Handle columns not in LLM response (use fallback)
        responded_names = {r.column_name for r in results}
        for col in columns:
            if col.column_name not in responded_names:
                results.append(self._fallback_single(col))

        return results, response.model

    def _deterministic_fallback(
        self, columns: list[ColumnAnalysisInput]
    ) -> list[ColumnAnalysisResult]:
        """Accept candidates as-is when LLM is unavailable."""
        return [self._fallback_single(col) for col in columns]

    def _fallback_single(self, col: ColumnAnalysisInput) -> ColumnAnalysisResult:
        """Produce a fallback result for a single column."""
        # If candidate has high confidence, confirm it deterministically
        if col.candidate_confidence >= 0.80:
            return ColumnAnalysisResult(
                column_name=col.column_name,
                decision=SchemaIntelligenceDecision.CONFIRMED,
                confirmed_semantic_type=col.candidate_semantic_type,
                confirmed_column_role=col.candidate_column_role,
                confidence=col.candidate_confidence * 0.9,  # Slight penalty for no LLM confirmation
                reasoning="deterministic_high_confidence_fallback",
            )

        return ColumnAnalysisResult(
            column_name=col.column_name,
            decision=SchemaIntelligenceDecision.UNRESOLVED,
            confirmed_semantic_type=col.candidate_semantic_type,
            confirmed_column_role=col.candidate_column_role,
            confidence=col.candidate_confidence,
            reasoning="fallback",
        )
