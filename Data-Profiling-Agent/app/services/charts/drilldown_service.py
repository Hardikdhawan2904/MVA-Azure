"""Drill-down service — serves pre-aggregated cube data."""

from typing import Any

from app.core.exceptions import ProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class DrillDownService:
    """Serves drill-down data from pre-aggregated cubes."""

    def execute_drill_down(
        self,
        cubes: list[dict[str, Any]],
        chart_id: str,
        hierarchy_levels: list[str],
        selected_path: dict[str, str],
    ) -> dict[str, Any]:
        """
        Execute a drill-down request against stored cubes.

        Validates:
        - Path columns belong to the hierarchy
        - Levels follow hierarchy order
        - Next level exists
        """
        # Validate selected_path against hierarchy
        for col in selected_path:
            if col not in hierarchy_levels:
                raise ProcessingError(
                    code="INVALID_DRILL_DOWN_PATH",
                    message=f"Column '{col}' is not part of the hierarchy.",
                    details={"column": col, "hierarchy": hierarchy_levels},
                )

        # Determine current depth and next level
        current_depth = len(selected_path)
        if current_depth >= len(hierarchy_levels):
            raise ProcessingError(
                code="DRILL_DOWN_LEAF_REACHED",
                message="Cannot drill down further — already at the deepest level.",
                details={"depth": current_depth, "max_levels": len(hierarchy_levels)},
            )

        current_level = hierarchy_levels[current_depth]
        next_level = hierarchy_levels[current_depth + 1] if current_depth + 1 < len(hierarchy_levels) else None

        # Find matching cube data
        matching_data: list[dict[str, Any]] = []
        for cube in cubes:
            if cube.get("level_column") != current_level:
                continue
            cube_path = cube.get("dimension_path_json") or {}
            # Check if cube path matches selected_path
            if all(cube_path.get(k) == v for k, v in selected_path.items()):
                matching_data.extend(cube.get("aggregated_data_json", []))

        return {
            "chart_id": chart_id,
            "current_level": current_level,
            "next_level": next_level,
            "selected_path": selected_path,
            "data": matching_data,
        }
