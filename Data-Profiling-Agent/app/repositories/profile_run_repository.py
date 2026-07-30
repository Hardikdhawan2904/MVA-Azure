"""Profile run repository — persists run results to PostgreSQL."""

import uuid
from typing import Any
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.profile_run import ProfileRun
from app.models.dataset_profile import DatasetProfile
from app.models.column_profile import ColumnProfile
from app.models.secondary_domain_result import SecondaryDomainResult
from app.models.column_classification import ColumnClassification
from app.models.hierarchy import HierarchyChain, HierarchyEdge
from app.models.quality_assessment import QualityAssessment
from app.models.readiness_assessment import ReadinessAssessment
from app.models.feature_recommendation import FeatureRecommendation
from app.models.chart_specification import ChartSpecification
from app.core.enums import RunStatus
from app.core.constants import PIPELINE_VERSION


class ProfileRunRepository:
    """Persists profiling results to PostgreSQL."""

    def __init__(self, session: Session):
        self._session = session

    def create_run(self, run_id: uuid.UUID, primary_domain: str,
                   filename: str, file_type: str) -> ProfileRun:
        """Create a new profile run record."""
        run = ProfileRun(
            id=run_id,
            status=RunStatus.PENDING.value,
            primary_domain=primary_domain,
            source_filename=filename,
            source_file_type=file_type,
            pipeline_version=PIPELINE_VERSION,
        )
        self._session.add(run)
        self._session.flush()
        return run

    def mark_processing(self, run_id: uuid.UUID) -> None:
        """Mark run as processing."""
        run = self._session.get(ProfileRun, run_id)
        if run:
            run.status = RunStatus.PROCESSING.value
            run.started_at = datetime.now(timezone.utc)
            self._session.flush()

    def mark_completed(self, run_id: uuid.UUID, row_count: int, column_count: int,
                       secondary_domain: str | None = None) -> None:
        """Mark run as completed."""
        run = self._session.get(ProfileRun, run_id)
        if run:
            run.status = RunStatus.COMPLETED.value
            run.completed_at = datetime.now(timezone.utc)
            run.row_count = row_count
            run.column_count = column_count
            run.dominant_secondary_domain = secondary_domain
            self._session.flush()

    def mark_failed(self, run_id: uuid.UUID, error_code: str, error_message: str) -> None:
        """Mark run as failed."""
        run = self._session.get(ProfileRun, run_id)
        if run:
            run.status = RunStatus.FAILED.value
            run.failed_at = datetime.now(timezone.utc)
            run.error_code = error_code
            run.error_message = error_message[:1000]
            self._session.flush()

    def get_run(self, run_id: uuid.UUID) -> ProfileRun | None:
        """Get a profile run by ID."""
        return self._session.get(ProfileRun, run_id)

    def persist_dataset_profile(self, run_id: uuid.UUID, profile_data: dict[str, Any]) -> None:
        """Persist dataset-level profile."""
        dp = DatasetProfile(
            run_id=run_id,
            row_count=profile_data.get("row_count", 0),
            column_count=profile_data.get("column_count", 0),
            duplicate_row_count=profile_data.get("duplicate_row_count", 0),
            memory_estimate_bytes=profile_data.get("memory_estimate_bytes"),
            inferred_grain=profile_data.get("inferred_grain"),
            profile_json=profile_data,
        )
        self._session.add(dp)

    def persist_column_profiles(self, run_id: uuid.UUID, columns: list[dict[str, Any]]) -> dict[str, uuid.UUID]:
        """Persist column profiles. Returns a map of physical_name -> generated column_profile id."""
        ids_by_name: dict[str, uuid.UUID] = {}
        for col in columns:
            cp = ColumnProfile(
                run_id=run_id,
                physical_name=col["physical_name"],
                normalized_key=col["normalized_key"],
                pandas_dtype=col.get("pandas_dtype", "object"),
                refined_data_type=col.get("refined_data_type", "unknown"),
                statistics_json=col.get("statistics"),
                candidate_semantic_type=col.get("candidate_semantic_type"),
                candidate_column_role=col.get("candidate_column_role"),
                candidate_confidence=col.get("candidate_confidence"),
                confirmed_semantic_type=col.get("confirmed_semantic_type"),
                confirmed_column_role=col.get("confirmed_column_role"),
                schema_confidence=col.get("schema_confidence"),
                identifier_score=col.get("identifier_score"),
                is_grain_key=col.get("is_grain_key", False),
            )
            self._session.add(cp)
            self._session.flush()
            ids_by_name[cp.physical_name] = cp.id
        return ids_by_name

    def persist_secondary_domain_result(self, run_id: uuid.UUID, secondary_domain: dict[str, Any]) -> None:
        """Persist the primary (rank=1) secondary-domain classification result."""
        sdr = SecondaryDomainResult(
            run_id=run_id,
            domain_name=secondary_domain.get("name"),
            rank=1,
            confidence=secondary_domain.get("confidence", 0.0),
            status=secondary_domain.get("status", "unresolved"),
            evidence_json=secondary_domain.get("evidence"),
        )
        self._session.add(sdr)

    def persist_column_classifications(
        self,
        run_id: uuid.UUID,
        column_profile_ids_by_name: dict[str, uuid.UUID],
        classifications: list[dict[str, Any]],
    ) -> None:
        """Persist per-column data-category classifications, linked to their column_profile row."""
        for c in classifications:
            column_profile_id = column_profile_ids_by_name.get(c.get("column_name"))
            if column_profile_id is None:
                continue
            cc = ColumnClassification(
                run_id=run_id,
                column_profile_id=column_profile_id,
                primary_category=c["primary_category"],
                secondary_categories_json=c.get("secondary_categories"),
                confidence=c.get("confidence", 0.0),
                evidence_json=c.get("evidence"),
            )
            self._session.add(cc)

    def persist_hierarchy(self, run_id: uuid.UUID, hierarchy: dict[str, Any]) -> None:
        """Persist the hierarchy chain and its edges."""
        chain = HierarchyChain(
            run_id=run_id,
            template_key=hierarchy.get("template_key"),
            status=hierarchy.get("status", "unresolved"),
            average_confidence=hierarchy.get("average_confidence"),
            level_columns_json=hierarchy.get("level_columns"),
            warnings_json=hierarchy.get("warnings"),
        )
        self._session.add(chain)
        self._session.flush()

        for edge in hierarchy.get("edges", []):
            he = HierarchyEdge(
                chain_id=chain.id,
                parent_column=edge["parent_column"],
                child_column=edge["child_column"],
                distinct_child_count=edge["distinct_child_count"],
                mapped_child_count=edge["mapped_child_count"],
                violating_child_count=edge["violating_child_count"],
                fd_consistency=edge["fd_consistency"],
                mapping_coverage=edge["mapping_coverage"],
                edge_confidence=edge["edge_confidence"],
                status=edge["status"],
                conflict_samples_json=edge.get("conflict_samples"),
            )
            self._session.add(he)

    def persist_quality_assessments(self, run_id: uuid.UUID, assessments: list[dict[str, Any]]) -> None:
        """Persist quality dimension assessments."""
        for a in assessments:
            qa = QualityAssessment(
                run_id=run_id,
                dimension=a["dimension"],
                score=a.get("score"),
                display_score=round(a["score"] * 100, 2) if a.get("score") is not None else None,
                status=a["status"],
                assessed_count=a.get("assessed_count"),
                violation_count=a.get("violation_count"),
                evidence_json=a.get("evidence"),
                reason=a.get("reason"),
            )
            self._session.add(qa)

    def persist_readiness_assessments(self, run_id: uuid.UUID, assessments: list[dict[str, Any]]) -> None:
        """Persist readiness assessments."""
        for a in assessments:
            ra = ReadinessAssessment(
                run_id=run_id,
                assessment_type=a["assessment_type"],
                score=a.get("score"),
                status=a["status"],
                strengths_json=a.get("strengths"),
                blockers_json=a.get("blocking_issues"),
                recommendations_json=a.get("recommendations"),
                evidence_json=a.get("evidence"),
                weight_profile_version=a.get("weight_profile_version"),
                dataset_score=a.get("dataset_score"),
                task_compatibility_score=a.get("task_compatibility_score"),
            )
            self._session.add(ra)

    def persist_feature_recommendation(
        self, run_id: uuid.UUID, recommendation: dict[str, Any], business_question: str | None = None
    ) -> None:
        """Persist the target/feature/drop recommendation for a run. A no-op
        when recommendation is empty (e.g. a lightweight/legacy result)."""
        if not recommendation:
            return
        fr = FeatureRecommendation(
            run_id=run_id,
            business_question=business_question,
            target_column=recommendation.get("target_column"),
            problem_type=recommendation.get("problem_type"),
            time_column=recommendation.get("time_column"),
            recommended_approach=recommendation.get("recommended_approach"),
            approach_reasoning=recommendation.get("approach_reasoning"),
            feature_columns_json=recommendation.get("feature_columns"),
            drop_columns_json=recommendation.get("drop_columns"),
            confidence=recommendation.get("confidence"),
        )
        self._session.add(fr)

    def persist_charts(self, run_id: uuid.UUID, charts: list[dict[str, Any]]) -> None:
        """Persist chart specifications."""
        for c in charts:
            cs = ChartSpecification(
                run_id=run_id,
                chart_key=c["chart_key"],
                category=c["category"],
                chart_type=c["chart_type"],
                title=c["title"],
                specification_json=c,
                aggregated_data_json=c.get("data"),
                rank=c.get("rank", 1),
            )
            self._session.add(cs)

    def get_dataset_profile(self, run_id: uuid.UUID) -> DatasetProfile | None:
        """Get the dataset-level profile for a run."""
        stmt = select(DatasetProfile).where(DatasetProfile.run_id == run_id)
        return self._session.execute(stmt).scalars().first()

    def list_column_profiles(self, run_id: uuid.UUID) -> list[ColumnProfile]:
        """List all column profiles for a run."""
        stmt = select(ColumnProfile).where(ColumnProfile.run_id == run_id)
        return list(self._session.execute(stmt).scalars().all())

    def list_column_classifications(self, run_id: uuid.UUID) -> list[ColumnClassification]:
        """List all column classifications for a run."""
        stmt = select(ColumnClassification).where(ColumnClassification.run_id == run_id)
        return list(self._session.execute(stmt).scalars().all())

    def get_secondary_domain_result(self, run_id: uuid.UUID) -> SecondaryDomainResult | None:
        """Get the primary (rank=1) secondary-domain result for a run."""
        stmt = select(SecondaryDomainResult).where(
            SecondaryDomainResult.run_id == run_id
        ).order_by(SecondaryDomainResult.rank)
        return self._session.execute(stmt).scalars().first()

    def get_hierarchy(self, run_id: uuid.UUID) -> dict[str, Any] | None:
        """Get the hierarchy chain and its edges for a run, assembled into the same shape as HierarchyChainResult.to_dict()."""
        chain_stmt = select(HierarchyChain).where(HierarchyChain.run_id == run_id)
        chain = self._session.execute(chain_stmt).scalars().first()
        if chain is None:
            return None

        edges_stmt = select(HierarchyEdge).where(HierarchyEdge.chain_id == chain.id)
        edges = list(self._session.execute(edges_stmt).scalars().all())

        return {
            "status": chain.status,
            "template_key": chain.template_key,
            "level_columns": chain.level_columns_json or [],
            "edge_count": len(edges),
            "edges": [
                {
                    "parent_column": e.parent_column,
                    "child_column": e.child_column,
                    "distinct_child_count": e.distinct_child_count,
                    "mapped_child_count": e.mapped_child_count,
                    "violating_child_count": e.violating_child_count,
                    "fd_consistency": e.fd_consistency,
                    "mapping_coverage": e.mapping_coverage,
                    "edge_confidence": e.edge_confidence,
                    "status": e.status,
                    "conflict_count": e.violating_child_count,
                    "conflict_samples": e.conflict_samples_json,
                }
                for e in edges
            ],
            "average_confidence": chain.average_confidence,
            "warnings": chain.warnings_json or [],
        }

    def list_quality_assessments(self, run_id: uuid.UUID) -> list[QualityAssessment]:
        """List all quality assessments for a run."""
        stmt = select(QualityAssessment).where(QualityAssessment.run_id == run_id)
        return list(self._session.execute(stmt).scalars().all())

    def list_readiness_assessments(self, run_id: uuid.UUID) -> list[ReadinessAssessment]:
        """List all readiness assessments for a run."""
        stmt = select(ReadinessAssessment).where(ReadinessAssessment.run_id == run_id)
        return list(self._session.execute(stmt).scalars().all())

    def list_charts(self, run_id: uuid.UUID) -> list[ChartSpecification]:
        """List all chart specifications for a run."""
        stmt = select(ChartSpecification).where(ChartSpecification.run_id == run_id)
        return list(self._session.execute(stmt).scalars().all())

    def get_feature_recommendation(self, run_id: uuid.UUID) -> FeatureRecommendation | None:
        """Get the target/feature/drop recommendation for a run."""
        stmt = select(FeatureRecommendation).where(FeatureRecommendation.run_id == run_id)
        return self._session.execute(stmt).scalars().first()

    def commit(self) -> None:
        """Commit the current transaction."""
        self._session.commit()
