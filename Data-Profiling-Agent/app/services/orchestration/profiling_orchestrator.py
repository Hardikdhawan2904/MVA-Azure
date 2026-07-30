"""Profiling orchestrator — runs the full pipeline end-to-end."""

import uuid
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.enums import (
    RunStatus, FileType, ColumnRole, HierarchyChainStatus, RuleSource,
)
from app.core.exceptions import UnsupportedDomainError
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.ingestion.file_validator import FileValidator
from app.services.ingestion.csv_loader import CSVLoader
from app.services.ingestion.xlsx_loader import XLSXLoader
from app.services.ingestion.temporary_storage import TemporaryStorage
from app.services.profiling.dataset_profiler import DatasetProfiler
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.identifier_detector import IdentifierDetector
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator
from app.services.schema_intelligence.interface import SchemaIntelligenceProvider
from app.services.schema_intelligence.models import ColumnAnalysisInput, DomainContext
from app.services.domains.secondary_domain_classifier import SecondaryDomainClassifier
from app.services.classification.data_category_classifier import DataCategoryClassifier
from app.services.hierarchy.candidate_builder import HierarchyCandidateBuilder
from app.services.rules.rule_loader import RuleLoader
from app.services.rules.rule_engine import RuleEngine
from app.services.rules.suggestion_generator import RuleSuggestionGenerator
from app.services.quality.completeness import assess_completeness
from app.services.quality.uniqueness import assess_uniqueness
from app.services.quality.validity import assess_validity
from app.services.quality.conformity import assess_conformity
from app.services.quality.consistency import assess_consistency
from app.services.quality.timeliness import assess_timeliness
from app.services.quality.integrity import assess_integrity
from app.services.quality.accuracy import assess_accuracy
from app.services.quality.business_rule_compliance import assess_business_rule_compliance
from app.services.quality.semantic_quality import assess_semantic_quality
from app.services.quality.overall_score import calculate_overall_score
from app.services.readiness.readiness_engine import ReadinessEngine
from app.services.charts.candidate_generator import ChartCandidateGenerator
from app.services.charts.aggregation_engine import AggregationEngine

logger = get_logger(__name__)


class PipelineResult:
    """Complete result of the profiling pipeline."""

    def __init__(self):
        self.run_id: str = ""
        self.status: RunStatus = RunStatus.PENDING
        self.primary_domain: str = ""
        self.secondary_domain: dict[str, Any] = {}
        self.dataset_profile: dict[str, Any] = {}
        self.column_profiles: list[dict[str, Any]] = []
        self.column_classifications: list[dict[str, Any]] = []
        self.hierarchy: dict[str, Any] = {}
        self.quality_assessments: list[dict[str, Any]] = []
        self.overall_quality: dict[str, Any] = {}
        self.readiness_assessments: list[dict[str, Any]] = []
        self.feature_recommendation: dict[str, Any] = {}
        self.charts: list[dict[str, Any]] = []
        self.drill_down_cubes: list[dict[str, Any]] = []
        self.rule_evaluations: list[dict[str, Any]] = []
        self.rule_suggestions: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: dict[str, Any] | None = None


class ProfilingOrchestrator:
    """Orchestrates the entire profiling pipeline."""

    def __init__(
        self,
        settings: Settings,
        config_repo: ConfigurationRepository,
        schema_intelligence: SchemaIntelligenceProvider,
        temp_storage: TemporaryStorage,
    ):
        self._settings = settings
        self._config_repo = config_repo
        self._schema_intelligence = schema_intelligence
        self._temp_storage = temp_storage
        self._file_validator = FileValidator(settings)
        self._csv_loader = CSVLoader(settings)
        self._xlsx_loader = XLSXLoader(settings)
        self._dataset_profiler = DatasetProfiler()
        self._column_profiler = ColumnProfiler(settings)
        self._type_refiner = TypeRefiner()
        self._identifier_detector = IdentifierDetector()
        self._semantic_generator = SemanticCandidateGenerator()
        self._domain_classifier = SecondaryDomainClassifier(config_repo)
        self._category_classifier = DataCategoryClassifier(
            weights=config_repo.get_classification_weights()
        )
        self._hierarchy_builder = HierarchyCandidateBuilder(config_repo)
        self._rule_loader = RuleLoader(config_repo)
        self._rule_engine = RuleEngine()
        self._readiness_engine = ReadinessEngine()
        self._chart_generator = ChartCandidateGenerator()
        self._agg_engine = AggregationEngine()

        llm_provider = schema_intelligence._llm if hasattr(schema_intelligence, '_llm') else None
        self._suggestion_generator = RuleSuggestionGenerator(llm_provider) if llm_provider else None

    def execute(
        self,
        run_id: uuid.UUID,
        file_content: bytes,
        filename: str,
        primary_domain: str,
        schema_metadata: dict[str, Any] | None = None,
        request_rules: list[dict[str, Any]] | None = None,
        sheet_name: str | None = None,
        approved_rule_dicts: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        """Execute the full profiling pipeline."""
        result = PipelineResult()
        result.run_id = str(run_id)
        result.started_at = datetime.now(timezone.utc).isoformat()

        try:
            # Validate domain
            stage_start = time.time()
            supported = self._config_repo.get_supported_primary_domains()
            if primary_domain not in supported:
                raise UnsupportedDomainError(
                    code="UNSUPPORTED_DOMAIN",
                    message=f"Domain '{primary_domain}' is not supported.",
                    details={"domain": primary_domain, "supported": supported},
                )
            result.primary_domain = primary_domain
            self._log_stage(run_id, "input_validation", stage_start)

            # Validate and save file
            stage_start = time.time()
            file_type = self._file_validator.validate(filename, len(file_content))
            file_path = self._temp_storage.save_upload(run_id, file_content, filename)
            self._log_stage(run_id, "file_storage", stage_start)

            # Load DataFrame
            stage_start = time.time()
            if file_type == FileType.CSV:
                df = self._csv_loader.load(file_path)
            else:
                df = self._xlsx_loader.load(file_path, sheet_name=sheet_name)
            self._log_stage(run_id, "dataframe_loading", stage_start, row_count=len(df), column_count=len(df.columns))

            # Dataset profiling
            stage_start = time.time()
            ds_profile = self._dataset_profiler.profile(df)
            result.dataset_profile = ds_profile.to_dict()
            self._log_stage(run_id, "dataset_profiling", stage_start)

            # Column profiling
            stage_start = time.time()
            col_profiles = self._column_profiler.profile_all(df, ds_profile.normalized_keys)
            self._log_stage(run_id, "column_profiling", stage_start, column_count=len(col_profiles))

            # Type refinement
            stage_start = time.time()
            refined_types = [self._type_refiner.refine(p) for p in col_profiles]
            self._log_stage(run_id, "type_refinement", stage_start)

            # Identifier detection
            stage_start = time.time()
            grain_result = self._identifier_detector.detect(col_profiles, refined_types)
            grain_columns = grain_result.grain_columns
            self._log_stage(run_id, "identifier_detection", stage_start)

            # Semantic candidates
            stage_start = time.time()
            id_flags = [c.is_identifier for c in grain_result.identifier_candidates]
            sem_candidates = self._semantic_generator.generate_all(col_profiles, refined_types, id_flags)
            self._log_stage(run_id, "semantic_candidates", stage_start)

            # Schema Intelligence
            stage_start = time.time()
            si_inputs = [
                ColumnAnalysisInput(
                    column_name=p.physical_name,
                    normalized_key=p.normalized_key,
                    description=self._get_description(p.physical_name, schema_metadata),
                    refined_physical_type=refined_types[i].value,
                    statistics_summary=p.to_statistics_dict(),
                    representative_sample_values=p.representative_values[:5],
                    candidate_semantic_type=sem_candidates[i].candidate_semantic_type,
                    candidate_column_role=sem_candidates[i].candidate_column_role.value,
                    candidate_confidence=sem_candidates[i].candidate_confidence,
                    primary_domain=primary_domain,
                )
                for i, p in enumerate(col_profiles)
            ]
            domain_ctx = DomainContext(
                primary_domain=primary_domain,
                row_count=ds_profile.row_count,
                column_count=ds_profile.column_count,
            )
            si_result = self._schema_intelligence.analyze(si_inputs, domain_ctx)
            self._log_stage(run_id, "schema_intelligence", stage_start)

            # Build column profile output
            result.column_profiles = self._build_column_output(
                col_profiles, refined_types, sem_candidates, si_result, grain_result, schema_metadata
            )

            # Secondary domain classification
            sec_domain = self._domain_classifier.classify(primary_domain, col_profiles, sem_candidates)
            result.secondary_domain = sec_domain.to_dict()

            # Column data-category classification
            classifications = self._category_classifier.classify_all(col_profiles, sem_candidates, primary_domain)
            result.column_classifications = [c.to_dict() for c in classifications]

            # Hierarchy inference
            hierarchy_result = self._hierarchy_builder.build(
                df, primary_domain, sec_domain.name, col_profiles, sem_candidates, grain_columns
            )
            result.hierarchy = hierarchy_result.to_dict()
            hierarchy_levels = hierarchy_result.level_columns

            # Business rules
            rules = self._rule_loader.load_domain_rules(primary_domain, sec_domain.name)
            if request_rules:
                rules.extend(self._rule_loader.load_request_rules(request_rules, primary_domain))
            if approved_rule_dicts:
                rules.extend(self._rule_loader.load_request_rules(
                    approved_rule_dicts, primary_domain, source=RuleSource.APPROVED_SUGGESTION
                ))

            # Build role map for rule resolution
            role_map = self._build_role_map(sem_candidates, col_profiles)
            rule_results = self._rule_engine.evaluate(df, rules, role_map)
            result.rule_evaluations = [r.to_dict() for r in rule_results]

            # Determine mandatory/expected-unique columns
            mandatory_cols = self._get_mandatory_columns(col_profiles, schema_metadata)
            unique_cols = self._get_expected_unique_columns(col_profiles, schema_metadata)

            # Quality assessments
            quality_dims: list[dict[str, Any]] = []
            comp = assess_completeness(col_profiles, mandatory_cols)
            quality_dims.append({
                "dimension": comp.dimension, "score": comp.score,
                "status": comp.status.value, "assessed_count": comp.assessed_count,
                "violation_count": comp.violation_count, "evidence": comp.evidence, "reason": comp.reason,
            })
            quality_dims.append(assess_uniqueness(col_profiles, unique_cols))
            quality_dims.append(assess_validity(rule_results))
            quality_dims.append(assess_conformity(rule_results))
            quality_dims.append(assess_consistency(rule_results))
            quality_dims.append(assess_business_rule_compliance(rule_results))
            quality_dims.append(assess_timeliness())
            quality_dims.append(assess_integrity())
            quality_dims.append(assess_accuracy())

            # Semantic quality
            si_conf_avg = self._avg_confidence(si_result.column_results) if si_result.column_results else None
            class_conf_avg = sum(c.confidence for c in classifications) / len(classifications) if classifications else None
            sec_conf = sec_domain.confidence if sec_domain.name else None
            hier_conf = hierarchy_result.average_confidence if hierarchy_result.status != HierarchyChainStatus.UNRESOLVED else None
            quality_dims.append(assess_semantic_quality(si_conf_avg, class_conf_avg, sec_conf, hier_conf))

            result.quality_assessments = quality_dims

            # Rule suggestions (AI-proposed, require approval before they ever get evaluated)
            if self._suggestion_generator is not None:
                # A short cooldown between this call and the schema-intelligence LLM call
                # immediately before it avoids bursting two requests in quick succession
                # against the same rate-limited endpoint. Configurable/disableable via
                # LLM_BURST_COOLDOWN_SECONDS if the configured Azure deployment's quota
                # makes it unnecessary.
                if self._settings.llm_burst_cooldown_seconds > 0:
                    time.sleep(self._settings.llm_burst_cooldown_seconds)
                try:
                    result.rule_suggestions = [
                        s.to_dict()
                        for s in self._suggestion_generator.generate(col_profiles, primary_domain, sec_domain.name)
                    ]
                except Exception as e:
                    logger.warning("rule_suggestion_generation_failed", run_id=str(run_id), error=str(e))

            # Overall quality score
            weights_config = self._config_repo.get_quality_weights()
            result.overall_quality = calculate_overall_score(quality_dims, weights_config.get("weights", {}))

            # Readiness
            has_temporal = any(c.candidate_column_role == ColumnRole.TEMPORAL_DIMENSION for c in sem_candidates)
            metric_count = sum(1 for c in sem_candidates if c.candidate_column_role == ColumnRole.METRIC)
            dim_count = sum(1 for c in sem_candidates if c.candidate_column_role == ColumnRole.DIMENSION)
            desc_coverage = self._calc_description_coverage(col_profiles, schema_metadata)

            readiness_results = self._readiness_engine.assess_all(
                col_profiles, quality_dims, grain_columns,
                has_temporal, metric_count, dim_count, desc_coverage, ds_profile.row_count,
            )
            result.readiness_assessments = [r.to_dict() for r in readiness_results]

            # Charts
            domain_config = self._config_repo.get_domain_config(primary_domain)
            chart_templates = domain_config.get("chart_templates", [])
            charts = self._chart_generator.generate(
                col_profiles, sem_candidates, chart_templates, quality_dims, hierarchy_levels
            )
            charts = self._agg_engine.aggregate_all(df, charts)
            result.charts = [c.to_dict() for c in charts]

            # Generate drill-down cubes (before DataFrame is discarded)
            stage_start = time.time()
            result.drill_down_cubes = self._generate_drill_down_cubes(
                df, charts, hierarchy_levels, sem_candidates, col_profiles
            )
            self._log_stage(run_id, "drill_down_cubes", stage_start)

            result.status = RunStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            result.status = RunStatus.FAILED
            result.error = {"code": getattr(e, "code", "PROCESSING_ERROR"), "message": str(e)}
            logger.error("pipeline_failed", run_id=str(run_id), error=str(e))
        finally:
            # Always clean up temp files
            self._temp_storage.cleanup_run(run_id)

        return result

    def _generate_drill_down_cubes(
        self,
        df: pd.DataFrame,
        charts: list,
        hierarchy_levels: list[str],
        sem_candidates: list,
        col_profiles: list,
    ) -> list[dict[str, Any]]:
        """
        Pre-aggregate drill-down cubes for hierarchy-enabled charts.

        Only generates cubes for charts that have hierarchy info and valid levels.
        Applies MIN_CUBE_GROUP_SIZE suppression.
        """
        if not hierarchy_levels or len(hierarchy_levels) < 2:
            return []

        min_group = self._settings.min_cube_group_size
        cubes: list[dict[str, Any]] = []

        # Find metric columns for aggregation
        metric_cols = [
            col_profiles[i].physical_name
            for i, c in enumerate(sem_candidates)
            if c.candidate_column_role == ColumnRole.METRIC
            and col_profiles[i].physical_name in df.columns
        ]

        if not metric_cols:
            return []

        primary_metric = metric_cols[0]
        numeric_metric = pd.to_numeric(df[primary_metric], errors="coerce")

        # Generate cubes for each hierarchy level depth
        for depth in range(len(hierarchy_levels)):
            level_col = hierarchy_levels[depth]
            if level_col not in df.columns:
                continue

            # Build dimension path (all levels above current)
            path_cols = hierarchy_levels[:depth]

            if not path_cols:
                # Top level — aggregate by this level only
                grouped = df.assign(_metric=numeric_metric).groupby(level_col)["_metric"]
                agg_data = []
                for label, group_metric in grouped:
                    count = int(group_metric.count())
                    if count < min_group:
                        continue
                    agg_data.append({
                        "label": str(label),
                        "value": round(float(group_metric.sum()), 2) if not group_metric.isna().all() else 0,
                        "count": count,
                    })

                if agg_data:
                    cubes.append({
                        "level_column": level_col,
                        "level_depth": depth,
                        "dimension_path_json": {},
                        "aggregated_data_json": agg_data[:100],
                        "metric_column": primary_metric,
                        "aggregation": "sum",
                        "record_count": len(df),
                    })
            else:
                # Deeper levels — aggregate for each unique parent path
                valid_path_cols = [c for c in path_cols if c in df.columns]
                if not valid_path_cols:
                    continue

                # Get unique parent combinations (bounded)
                parent_combos = df[valid_path_cols].drop_duplicates().head(50)

                for _, parent_row in parent_combos.iterrows():
                    mask = pd.Series(True, index=df.index)
                    path_dict: dict[str, str] = {}
                    for pc in valid_path_cols:
                        mask &= df[pc] == parent_row[pc]
                        path_dict[pc] = str(parent_row[pc])

                    subset = df[mask]
                    if len(subset) < min_group:
                        continue

                    sub_metric = pd.to_numeric(subset[primary_metric], errors="coerce")
                    grouped = subset.assign(_m=sub_metric).groupby(level_col)["_m"]

                    agg_data = []
                    for label, group_metric in grouped:
                        count = int(group_metric.count())
                        if count < min_group:
                            continue
                        agg_data.append({
                            "label": str(label),
                            "value": round(float(group_metric.sum()), 2) if not group_metric.isna().all() else 0,
                            "count": count,
                        })

                    if agg_data:
                        cubes.append({
                            "level_column": level_col,
                            "level_depth": depth,
                            "dimension_path_json": path_dict,
                            "aggregated_data_json": agg_data[:100],
                            "metric_column": primary_metric,
                            "aggregation": "sum",
                            "record_count": len(subset),
                        })

        return cubes

    def _get_description(self, col_name: str, metadata: dict[str, Any] | None) -> str | None:
        if not metadata or "columns" not in metadata:
            return None
        for col in metadata["columns"]:
            if col.get("column_name") == col_name:
                return col.get("description")
        return None

    def _build_column_output(self, profiles, refined_types, candidates, si_result, grain_result, metadata):
        output = []
        si_map = {r.column_name: r for r in si_result.column_results}
        id_map = {c.column_name: c for c in grain_result.identifier_candidates}
        for i, p in enumerate(profiles):
            si = si_map.get(p.physical_name)
            id_c = id_map.get(p.physical_name)
            output.append({
                "physical_name": p.physical_name,
                "normalized_key": p.normalized_key,
                "pandas_dtype": p.pandas_dtype,
                "refined_data_type": refined_types[i].value,
                "statistics": p.to_statistics_dict(),
                "candidate_semantic_type": candidates[i].candidate_semantic_type,
                "candidate_column_role": candidates[i].candidate_column_role.value,
                "candidate_confidence": candidates[i].candidate_confidence,
                "confirmed_semantic_type": si.confirmed_semantic_type if si else None,
                "confirmed_column_role": si.confirmed_column_role if si else None,
                "schema_intelligence_decision": si.decision.value if si else None,
                "schema_confidence": si.confidence if si else None,
                "identifier_score": id_c.identifier_score if id_c else None,
                "is_grain_key": p.physical_name in grain_result.grain_columns,
            })
        return output

    def _build_role_map(self, candidates, profiles) -> dict[str, str]:
        role_map: dict[str, str] = {}
        for i, c in enumerate(candidates):
            if c.candidate_semantic_type:
                role_map[c.candidate_semantic_type] = profiles[i].physical_name
        return role_map

    def _get_mandatory_columns(self, profiles, metadata) -> list[str]:
        if metadata and "columns" in metadata:
            return [c["column_name"] for c in metadata["columns"] if c.get("mandatory")]
        return []

    def _get_expected_unique_columns(self, profiles, metadata) -> list[str]:
        if metadata and "columns" in metadata:
            return [c["column_name"] for c in metadata["columns"] if c.get("expected_unique")]
        return []

    def _calc_description_coverage(self, profiles, metadata) -> float:
        if not metadata or "columns" not in metadata:
            return 0.0
        described = sum(1 for c in metadata["columns"] if c.get("description"))
        return described / max(len(profiles), 1)

    def _avg_confidence(self, results) -> float:
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)

    def _log_stage(self, run_id: uuid.UUID, stage: str, start_time: float, **kwargs) -> None:
        """Log a pipeline stage completion with timing."""
        duration_ms = round((time.time() - start_time) * 1000, 1)
        logger.info(
            "pipeline_stage",
            run_id=str(run_id),
            stage=stage,
            status="success",
            duration_ms=duration_ms,
            **kwargs,
        )
