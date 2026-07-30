"""Secondary-domain classifier — deterministic-first with optional LLM fallback."""

from typing import Any

from app.core.constants import (
    SECONDARY_DOMAIN_CLASSIFIED_THRESHOLD,
    SECONDARY_DOMAIN_NEEDS_REVIEW_THRESHOLD,
)
from app.core.enums import SecondaryDomainStatus
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.domains.evidence_builder import EvidenceBuilder
from app.services.llm.interface import LLMProvider, LLMRequest
from app.services.llm.prompts import SECONDARY_DOMAIN_SYSTEM, SECONDARY_DOMAIN_PROMPT_V1
from app.services.llm.structured_output import SecondaryDomainDecision
from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate

logger = get_logger(__name__)


class SecondaryDomainResult:
    """Result of secondary-domain classification."""

    def __init__(
        self,
        name: str | None,
        confidence: float,
        status: SecondaryDomainStatus,
        candidates: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ):
        self.name = name
        self.confidence = confidence
        self.status = status
        self.candidates = candidates
        self.evidence = evidence or []

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
        }
        if self.candidates:
            result["candidates"] = self.candidates
        if self.evidence:
            result["evidence"] = self.evidence
        return result


class SecondaryDomainClassifier:
    """
    Classifies datasets into a secondary domain.

    Process:
    1. Load allowed secondary domains from configuration.
    2. Build deterministic evidence scores.
    3. If top score >= classified threshold → classified.
    4. If ambiguous (needs_review range) → optionally use LLM.
    5. Constrain LLM to configured values only.
    """

    def __init__(
        self,
        config_repo: ConfigurationRepository,
        llm_provider: LLMProvider | None = None,
        classified_threshold: float = SECONDARY_DOMAIN_CLASSIFIED_THRESHOLD,
        needs_review_threshold: float = SECONDARY_DOMAIN_NEEDS_REVIEW_THRESHOLD,
    ):
        self._config_repo = config_repo
        self._llm = llm_provider
        self._classified_threshold = classified_threshold
        self._needs_review_threshold = needs_review_threshold
        self._evidence_builder = EvidenceBuilder()

    def classify(
        self,
        primary_domain: str,
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
    ) -> SecondaryDomainResult:
        """
        Classify the dataset into a secondary domain.

        Returns exactly one result with status classified, needs_review, or unresolved.
        """
        # Load allowed secondary domains
        secondary_config = self._config_repo.get_secondary_domains_for(primary_domain)
        if not secondary_config:
            return SecondaryDomainResult(
                name=None,
                confidence=0.0,
                status=SecondaryDomainStatus.UNRESOLVED,
                evidence=[{"signal": "no_secondary_domains_configured"}],
            )

        # Build deterministic scores
        scored = self._evidence_builder.build_scores(
            secondary_config, profiles, semantic_candidates
        )

        if not scored:
            return SecondaryDomainResult(
                name=None,
                confidence=0.0,
                status=SecondaryDomainStatus.UNRESOLVED,
            )

        top = scored[0]
        top_confidence = top.score

        # Case 1: High confidence — classified
        if top_confidence >= self._classified_threshold:
            return SecondaryDomainResult(
                name=top.domain_name,
                confidence=top_confidence,
                status=SecondaryDomainStatus.CLASSIFIED,
                evidence=[e.to_dict() for e in top.evidence],
            )

        # Case 2: Ambiguous — try LLM if available
        if top_confidence >= self._needs_review_threshold and self._llm:
            llm_result = self._try_llm_classification(
                primary_domain, secondary_config, profiles, semantic_candidates
            )
            if llm_result:
                return llm_result

        # Case 3: Needs review or unresolved based on threshold
        if top_confidence >= self._needs_review_threshold:
            candidates = [
                {"name": s.domain_name, "confidence": round(s.score, 4)}
                for s in scored[:3]
                if s.score >= self._needs_review_threshold
            ]
            return SecondaryDomainResult(
                name=None,
                confidence=top_confidence,
                status=SecondaryDomainStatus.NEEDS_REVIEW,
                candidates=candidates,
                evidence=[e.to_dict() for e in top.evidence],
            )

        # Case 4: Below threshold — unresolved
        candidates = [
            {"name": s.domain_name, "confidence": round(s.score, 4)}
            for s in scored[:3]
            if s.score > 0
        ]
        return SecondaryDomainResult(
            name=None,
            confidence=top_confidence,
            status=SecondaryDomainStatus.UNRESOLVED,
            candidates=candidates if candidates else None,
        )

    def _try_llm_classification(
        self,
        primary_domain: str,
        secondary_config: dict[str, Any],
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
    ) -> SecondaryDomainResult | None:
        """Attempt LLM-assisted classification for ambiguous cases."""
        allowed_domains = list(secondary_config.keys())
        column_names = [p.physical_name for p in profiles[:30]]
        semantic_types = [
            c.candidate_semantic_type for c in semantic_candidates[:30]
            if c.candidate_semantic_type
        ]
        column_roles = [
            c.candidate_column_role.value for c in semantic_candidates[:30]
        ]
        sample_values = {}
        for p in profiles[:10]:
            if p.representative_values:
                sample_values[p.physical_name] = p.representative_values[:3]

        prompt = SECONDARY_DOMAIN_PROMPT_V1.format(
            primary_domain=primary_domain,
            allowed_domains=", ".join(allowed_domains),
            column_names=", ".join(column_names),
            semantic_types=", ".join(semantic_types[:15]),
            column_roles=", ".join(column_roles[:15]),
            sample_values=str(sample_values)[:500],
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=SECONDARY_DOMAIN_SYSTEM,
            temperature=0.1,
            max_tokens=500,
        )

        parsed, response = self._llm.complete_structured(request, SecondaryDomainDecision)
        if parsed is None:
            logger.warning("secondary_domain_llm_failed", error=response.error)
            return None

        # Validate that LLM selected from allowed domains
        if parsed.selected_domain and parsed.selected_domain not in allowed_domains:
            logger.warning(
                "secondary_domain_llm_invalid",
                selected=parsed.selected_domain,
                allowed=allowed_domains,
            )
            return None

        if parsed.selected_domain and parsed.confidence >= self._classified_threshold:
            return SecondaryDomainResult(
                name=parsed.selected_domain,
                confidence=parsed.confidence,
                status=SecondaryDomainStatus.CLASSIFIED,
                evidence=[{"signal": "llm_classification", "reasoning": parsed.reasoning}],
            )

        if parsed.selected_domain and parsed.confidence >= self._needs_review_threshold:
            return SecondaryDomainResult(
                name=None,
                confidence=parsed.confidence,
                status=SecondaryDomainStatus.NEEDS_REVIEW,
                candidates=[{"name": parsed.selected_domain, "confidence": parsed.confidence}],
                evidence=[{"signal": "llm_classification", "reasoning": parsed.reasoning}],
            )

        return None
