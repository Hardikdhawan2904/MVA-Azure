"""Schema Intelligence provider interface (Protocol)."""

from typing import Protocol

from app.services.schema_intelligence.models import (
    ColumnAnalysisInput,
    DomainContext,
    SchemaIntelligenceResult,
)


class SchemaIntelligenceProvider(Protocol):
    """
    Protocol for Schema Intelligence implementations.

    Responsibilities:
    - Confirm or override candidate semantic types
    - Confirm or override candidate column roles
    - Recommend mandatory or expected-unique flags
    - Propose business rules
    - Assist with secondary-domain classification

    Must NOT infer physical data types.
    """

    def analyze(
        self,
        columns: list[ColumnAnalysisInput],
        domain_context: DomainContext,
    ) -> SchemaIntelligenceResult:
        """
        Analyze columns and return semantic intelligence results.

        Args:
            columns: List of column analysis inputs with candidates.
            domain_context: Domain information for contextual decisions.

        Returns:
            SchemaIntelligenceResult with per-column decisions.
        """
        ...
