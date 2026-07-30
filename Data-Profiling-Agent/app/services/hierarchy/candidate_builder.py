"""Candidate builder — orchestrates hierarchy inference end-to-end."""

import pandas as pd

from app.core.enums import HierarchyChainStatus
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.hierarchy.template_matcher import TemplateMatcher
from app.services.hierarchy.chain_selector import ChainSelector, HierarchyChainResult
from app.services.hierarchy.functional_dependency import FunctionalDependencyValidator
from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate

logger = get_logger(__name__)


class HierarchyCandidateBuilder:
    """
    Orchestrates single-hierarchy-chain inference.

    Pipeline:
    1. Load templates for domain + secondary domain
    2. Match template concepts to columns (exclude grain keys)
    3. Validate edges via functional dependencies
    4. Select best chain
    """

    def __init__(
        self,
        config_repo: ConfigurationRepository,
        fd_validator: FunctionalDependencyValidator | None = None,
    ):
        self._config_repo = config_repo
        self._matcher = TemplateMatcher()
        self._fd_validator = fd_validator or FunctionalDependencyValidator()
        self._chain_selector = ChainSelector(self._fd_validator)

    def build(
        self,
        df: pd.DataFrame,
        primary_domain: str,
        secondary_domain: str | None,
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
        grain_columns: list[str],
    ) -> HierarchyChainResult:
        """
        Build and validate a hierarchy chain for the dataset.

        Returns at most one chain or an unresolved result.
        """
        # Load templates from domain config
        domain_config = self._config_repo.get_domain_config(primary_domain)
        templates = domain_config.get("hierarchy_templates", [])

        if not templates:
            return HierarchyChainResult(
                status=HierarchyChainStatus.UNRESOLVED,
                template_key=None,
                level_columns=[],
                edges=[],
                average_confidence=0.0,
                warnings=["No hierarchy templates configured for domain"],
                rejected_edges=[],
            )

        # Match templates to columns
        matches = self._matcher.match_templates(
            templates, profiles, semantic_candidates, grain_columns
        )

        # Filter to templates with at least 2 matched levels
        viable_matches = [m for m in matches if len(m.matched_levels) >= 2]

        if not viable_matches:
            return HierarchyChainResult(
                status=HierarchyChainStatus.UNRESOLVED,
                template_key=None,
                level_columns=[],
                edges=[],
                average_confidence=0.0,
                warnings=["No template matched 2+ columns in the dataset"],
                rejected_edges=[],
            )

        # Select best chain with FD validation
        result = self._chain_selector.select(df, viable_matches)

        logger.info(
            "hierarchy_chain_selected",
            status=result.status.value,
            template_key=result.template_key,
            levels=len(result.level_columns),
            avg_confidence=round(result.average_confidence, 3),
        )

        return result
