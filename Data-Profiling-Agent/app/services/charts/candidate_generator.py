"""Chart candidate generator — creates chart specs from templates and available fields."""

from typing import Any

from app.core.enums import ChartType, ChartCategory, ColumnRole
from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate


class ChartSpec:
    """A generated chart specification with aggregated data."""

    def __init__(
        self,
        chart_key: str,
        category: ChartCategory,
        chart_type: ChartType,
        title: str,
        description: str = "",
        dimension: str | None = None,
        metric: str | None = None,
        aggregation: str = "count",
        data: list[dict[str, Any]] | None = None,
        hierarchy_info: dict[str, Any] | None = None,
        rank: int = 0,
        warnings: list[str] | None = None,
    ):
        self.chart_key = chart_key
        self.category = category
        self.chart_type = chart_type
        self.title = title
        self.description = description
        self.dimension = dimension
        self.metric = metric
        self.aggregation = aggregation
        self.data = data or []
        self.hierarchy_info = hierarchy_info
        self.rank = rank
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_key": self.chart_key,
            "category": self.category.value,
            "chart_type": self.chart_type.value,
            "title": self.title,
            "description": self.description,
            "dimension": self.dimension,
            "metric": self.metric,
            "aggregation": self.aggregation,
            "data": self.data,
            "hierarchy": self.hierarchy_info,
            "rank": self.rank,
            "warnings": self.warnings,
        }


class ChartCandidateGenerator:
    """Generates chart candidates from profiles and domain templates."""

    def __init__(self, max_charts: int = 5, target_business: int = 3, target_profiling: int = 2):
        self._max_charts = max_charts
        self._target_business = target_business
        self._target_profiling = target_profiling

    def generate(
        self,
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
        chart_templates: list[dict[str, Any]],
        quality_results: list[dict[str, Any]],
        hierarchy_levels: list[str] | None = None,
    ) -> list[ChartSpec]:
        """Generate up to max_charts chart specifications."""
        # Categorize available columns
        metrics = [
            (p, c) for p, c in zip(profiles, semantic_candidates)
            if c.candidate_column_role == ColumnRole.METRIC
        ]
        dimensions = [
            (p, c) for p, c in zip(profiles, semantic_candidates)
            if c.candidate_column_role == ColumnRole.DIMENSION
        ]
        temporals = [
            (p, c) for p, c in zip(profiles, semantic_candidates)
            if c.candidate_column_role == ColumnRole.TEMPORAL_DIMENSION
        ]

        charts: list[ChartSpec] = []

        # Generate business charts from templates
        business_charts = self._generate_business_charts(
            metrics, dimensions, temporals, chart_templates, hierarchy_levels
        )
        charts.extend(business_charts[:self._target_business])

        # Generate profiling/quality charts
        profiling_charts = self._generate_profiling_charts(profiles, quality_results)
        charts.extend(profiling_charts[:self._target_profiling])

        # Limit to max and assign ranks
        charts = charts[:self._max_charts]
        for i, chart in enumerate(charts):
            chart.rank = i + 1

        return charts

    def _generate_business_charts(
        self,
        metrics: list[tuple],
        dimensions: list[tuple],
        temporals: list[tuple],
        templates: list[dict[str, Any]],
        hierarchy_levels: list[str] | None,
    ) -> list[ChartSpec]:
        """Generate business charts — prefers domain-specific templates (matched
        against required_roles), tops up with the generic rules below so coverage
        never regresses below what those rules alone would produce (e.g. when a
        domain has no templates, or this dataset's columns don't satisfy any)."""
        charts: list[ChartSpec] = []

        if templates:
            charts.extend(self._resolve_templates(metrics, dimensions, temporals, templates, hierarchy_levels))

        if len(charts) < self._target_business:
            # Dedupe by what the chart actually shows, not just its key — a
            # template chart and a generic-rule chart can independently arrive
            # at the identical (chart_type, dimension, metric) combination
            # (e.g. "transaction_value_over_time" vs. the generic "X_over_time"
            # rule both plotting the same metric over the same temporal column).
            seen_signatures = {(c.chart_type, c.dimension, c.metric) for c in charts}
            for c in self._generate_generic_business_charts(metrics, dimensions, temporals, hierarchy_levels):
                signature = (c.chart_type, c.dimension, c.metric)
                if signature not in seen_signatures:
                    charts.append(c)
                    seen_signatures.add(signature)

        return charts

    def _resolve_templates(
        self,
        metrics: list[tuple],
        dimensions: list[tuple],
        temporals: list[tuple],
        templates: list[dict[str, Any]],
        hierarchy_levels: list[str] | None,
    ) -> list[ChartSpec]:
        """Match each domain-configured template's required_roles against this
        dataset's actual columns. A template only produces a chart when every
        required role resolves to at least one column."""
        all_cols = metrics + dimensions + temporals
        charts: list[ChartSpec] = []

        for template in templates:
            required_roles = template.get("required_roles", [])
            resolved: list[tuple[str, tuple]] = []
            unmatched = False

            for role in required_roles:
                match = self._resolve_role(role, all_cols, hierarchy_levels)
                if match is None:
                    unmatched = True
                    break
                resolved.append((role, match))

            if unmatched or not resolved:
                continue

            dimension_col: str | None = None
            metric_col: str | None = None
            for role, (p, c) in resolved:
                if role == "temporal_dimension" or role == "hierarchy_dimension":
                    if dimension_col is None:
                        dimension_col = p.physical_name
                elif c.candidate_column_role == ColumnRole.METRIC:
                    if metric_col is None:
                        metric_col = p.physical_name
                else:
                    if dimension_col is None:
                        dimension_col = p.physical_name

            hierarchy_info = None
            title = template.get("title_template", template["key"])
            if "hierarchy_dimension" in required_roles and hierarchy_levels:
                hierarchy_info = {
                    "current_level": hierarchy_levels[0],
                    "next_level": hierarchy_levels[1] if len(hierarchy_levels) > 1 else None,
                }
                title = title.replace("{hierarchy_level}", hierarchy_levels[0])

            charts.append(ChartSpec(
                chart_key=template["key"],
                category=ChartCategory(template.get("category", "business")),
                chart_type=ChartType(template["chart_type"]),
                title=title,
                dimension=dimension_col,
                metric=metric_col,
                aggregation=template.get("aggregation", "count"),
                hierarchy_info=hierarchy_info,
            ))

        return charts

    def _resolve_role(
        self, role: str, columns: list[tuple], hierarchy_levels: list[str] | None,
    ) -> tuple | None:
        """Find the first column satisfying a required_roles entry — checked
        against ColumnRole, then candidate_semantic_type, then two special
        cases that aren't per-column facts: hierarchy_dimension (the
        hierarchy's own top level — hierarchy_levels[0], see below) and
        status_dimension (alias for the generator's generic "status"
        default)."""
        if role == "hierarchy_dimension":
            if not hierarchy_levels:
                return None
            # Must be hierarchy_levels[0] specifically, not just "any
            # column in the hierarchy chain" -- _resolve_templates' title
            # substitution (a few lines below in _resolve_templates)
            # always uses hierarchy_levels[0] for "{hierarchy_level}",
            # so picking a different level here silently mismatches the
            # chart's title against its actual data. Confirmed live: with
            # hierarchy_levels=["region", "country_name"], iterating
            # `columns` in file order matched "country_name" first (it's
            # column 6 vs "region" at column 7 in that CSV) while the
            # title still said "region" -- a bar chart titled "Premium by
            # region" whose bars were actually 6 individual countries.
            target = hierarchy_levels[0]
            for p, c in columns:
                if p.physical_name == target:
                    return (p, c)
            return None

        for p, c in columns:
            if c.candidate_column_role.value == role:
                return (p, c)
            if c.candidate_semantic_type == role:
                return (p, c)
            if role == "status_dimension" and c.candidate_semantic_type == "status":
                return (p, c)
        return None

    def _generate_generic_business_charts(
        self,
        metrics: list[tuple],
        dimensions: list[tuple],
        temporals: list[tuple],
        hierarchy_levels: list[str] | None,
    ) -> list[ChartSpec]:
        """Generic, domain-agnostic fallback rules — used when a domain has no
        templates, or this dataset's columns don't satisfy any of them."""
        charts: list[ChartSpec] = []

        # Line chart: metric over time (if temporal + metric available)
        if temporals and metrics:
            temporal_col = temporals[0][0].physical_name
            metric_col = metrics[0][0].physical_name
            charts.append(ChartSpec(
                chart_key=f"{metric_col}_over_time",
                category=ChartCategory.BUSINESS,
                chart_type=ChartType.LINE,
                title=f"{metrics[0][0].physical_name} Over Time",
                dimension=temporal_col,
                metric=metric_col,
                aggregation="sum",
            ))

        # Bar chart: metric by dimension
        if dimensions and metrics:
            dim_col = dimensions[0][0].physical_name
            metric_col = metrics[0][0].physical_name
            if dimensions[0][0].distinct_count <= 30:
                charts.append(ChartSpec(
                    chart_key=f"{metric_col}_by_{dim_col}",
                    category=ChartCategory.BUSINESS,
                    chart_type=ChartType.BAR,
                    title=f"{metric_col} by {dim_col}",
                    dimension=dim_col,
                    metric=metric_col,
                    aggregation="sum",
                ))

        # Pie/donut: distribution of a dimension
        if dimensions:
            for dim_p, dim_c in dimensions[:2]:
                if 2 <= dim_p.distinct_count <= 10:
                    charts.append(ChartSpec(
                        chart_key=f"{dim_p.physical_name}_distribution",
                        category=ChartCategory.BUSINESS,
                        chart_type=ChartType.PIE,
                        title=f"{dim_p.physical_name} Distribution",
                        dimension=dim_p.physical_name,
                        aggregation="count",
                    ))
                    break

        # KPI: key metric summary
        if metrics:
            charts.append(ChartSpec(
                chart_key=f"{metrics[0][0].physical_name}_kpi",
                category=ChartCategory.BUSINESS,
                chart_type=ChartType.KPI,
                title=f"Total {metrics[0][0].physical_name}",
                metric=metrics[0][0].physical_name,
                aggregation="sum",
            ))

        # Hierarchy bar chart
        if hierarchy_levels and len(hierarchy_levels) >= 2 and metrics:
            charts.append(ChartSpec(
                chart_key=f"metric_by_{hierarchy_levels[0]}",
                category=ChartCategory.BUSINESS,
                chart_type=ChartType.BAR,
                title=f"{metrics[0][0].physical_name} by {hierarchy_levels[0]}",
                dimension=hierarchy_levels[0],
                metric=metrics[0][0].physical_name,
                aggregation="sum",
                hierarchy_info={
                    "current_level": hierarchy_levels[0],
                    "next_level": hierarchy_levels[1] if len(hierarchy_levels) > 1 else None,
                },
            ))

        return charts

    def _generate_profiling_charts(
        self, profiles: list[ColumnProfileResult], quality_results: list[dict[str, Any]]
    ) -> list[ChartSpec]:
        """Generate profiling and quality charts."""
        charts: list[ChartSpec] = []

        # Missing values by column
        cols_with_nulls = [(p.physical_name, p.null_ratio) for p in profiles if p.null_ratio > 0]
        if cols_with_nulls:
            cols_with_nulls.sort(key=lambda x: x[1], reverse=True)
            data = [{"label": c, "value": round(r * 100, 1)} for c, r in cols_with_nulls[:15]]
            charts.append(ChartSpec(
                chart_key="missing_values_by_column",
                category=ChartCategory.PROFILING,
                chart_type=ChartType.BAR,
                title="Missing Values by Column (%)",
                data=data,
                aggregation="value",
            ))

        # Quality scores by dimension
        assessed = [r for r in quality_results if r.get("status") == "assessed" and r.get("score") is not None]
        if assessed:
            data = [{"label": r["dimension"], "value": round(r["score"] * 100, 1)} for r in assessed]
            charts.append(ChartSpec(
                chart_key="quality_score_by_dimension",
                category=ChartCategory.QUALITY,
                chart_type=ChartType.BAR,
                title="Quality Score by Dimension",
                data=data,
                aggregation="value",
            ))

        return charts
