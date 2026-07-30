"""Deterministic rule engine — evaluates business rules against a DataFrame."""

from typing import Any

import pandas as pd

from app.core.enums import RuleType, RuleSource
from app.core.logging import get_logger
from app.services.rules.rule_loader import RuleDefinition

logger = get_logger(__name__)


class RuleEvaluationResult:
    """Result of evaluating a single rule against the dataset."""

    def __init__(
        self,
        rule_key: str,
        rule_type: RuleType,
        records_checked: int,
        pass_count: int,
        fail_count: int,
        score: float,
        target_columns: list[str],
        source: RuleSource,
        error: str | None = None,
    ):
        self.rule_key = rule_key
        self.rule_type = rule_type
        self.records_checked = records_checked
        self.pass_count = pass_count
        self.fail_count = fail_count
        self.score = score
        self.target_columns = target_columns
        self.source = source
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_key": self.rule_key,
            "rule_type": self.rule_type.value,
            "records_checked": self.records_checked,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "score": round(self.score, 4),
            "target_columns": self.target_columns,
            "source": self.source.value,
            "error": self.error,
        }


class RuleEngine:
    """
    Deterministic business rule engine.

    Evaluates typed rules against a pandas DataFrame.
    Does NOT execute arbitrary Python or SQL.
    Uses an allowlisted set of rule types.
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        rules: list[RuleDefinition],
        column_role_map: dict[str, str] | None = None,
    ) -> list[RuleEvaluationResult]:
        """
        Evaluate all active rules against the dataset.

        column_role_map maps semantic roles to actual column names.
        """
        role_map = column_role_map or {}
        results: list[RuleEvaluationResult] = []

        for rule in rules:
            if not rule.active:
                continue
            result = self._evaluate_single(df, rule, role_map)
            results.append(result)

        return results

    def _evaluate_single(
        self,
        df: pd.DataFrame,
        rule: RuleDefinition,
        role_map: dict[str, str],
    ) -> RuleEvaluationResult:
        """Evaluate a single rule."""
        try:
            if rule.rule_type == RuleType.NON_NULL:
                return self._eval_non_null(df, rule, role_map)
            elif rule.rule_type == RuleType.EXPECTED_UNIQUE:
                return self._eval_expected_unique(df, rule, role_map)
            elif rule.rule_type == RuleType.REGEX_MATCH:
                return self._eval_regex_match(df, rule, role_map)
            elif rule.rule_type == RuleType.ALLOWED_VALUES:
                return self._eval_allowed_values(df, rule, role_map)
            elif rule.rule_type == RuleType.NUMERIC_RANGE:
                return self._eval_numeric_range(df, rule, role_map)
            elif rule.rule_type == RuleType.COLUMN_COMPARISON:
                return self._eval_column_comparison(df, rule, role_map)
            elif rule.rule_type == RuleType.CONDITIONAL_REQUIRED:
                return self._eval_conditional_required(df, rule, role_map)
            else:
                return RuleEvaluationResult(
                    rule_key=rule.rule_key,
                    rule_type=rule.rule_type,
                    records_checked=0,
                    pass_count=0,
                    fail_count=0,
                    score=0.0,
                    target_columns=[],
                    source=rule.source,
                    error=f"Unsupported rule type: {rule.rule_type.value}",
                )
        except Exception as e:
            return RuleEvaluationResult(
                rule_key=rule.rule_key,
                rule_type=rule.rule_type,
                records_checked=0,
                pass_count=0,
                fail_count=0,
                score=0.0,
                target_columns=[],
                source=rule.source,
                error=str(e),
            )

    def _resolve_column(
        self, params: dict[str, Any], role_map: dict[str, str], df: pd.DataFrame
    ) -> str | None:
        """Resolve a column by direct name or role mapping."""
        col = params.get("target_column")
        if col and col in df.columns:
            return col
        role = params.get("target_role")
        if role and role in role_map:
            mapped = role_map[role]
            if mapped in df.columns:
                return mapped
        return None

    def _resolve_two_columns(
        self, params: dict, role_map: dict, df: pd.DataFrame, left_key: str, right_key: str
    ) -> tuple[str | None, str | None]:
        """Resolve left and right columns."""
        left = params.get(f"{left_key}_column")
        if not left or left not in df.columns:
            left_role = params.get(f"{left_key}_role")
            if left_role and left_role in role_map:
                left = role_map[left_role]
        right = params.get(f"{right_key}_column")
        if not right or right not in df.columns:
            right_role = params.get(f"{right_key}_role")
            if right_role and right_role in role_map:
                right = role_map[right_role]
        left_valid = left if left and left in df.columns else None
        right_valid = right if right and right in df.columns else None
        return left_valid, right_valid

    def _eval_non_null(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        col = self._resolve_column(rule.parameters, role_map, df)
        if not col:
            return self._skip_result(rule, "Column not found")
        total = len(df)
        fail_count = int(df[col].isna().sum())
        pass_count = total - fail_count
        score = pass_count / total if total > 0 else 0.0
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[col],
        )

    def _eval_expected_unique(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        col = self._resolve_column(rule.parameters, role_map, df)
        if not col:
            return self._skip_result(rule, "Column not found")
        total = len(df)
        duplicated = int(df[col].duplicated(keep="first").sum())
        pass_count = total - duplicated
        score = pass_count / total if total > 0 else 0.0
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=duplicated,
            score=score, target_columns=[col],
        )

    def _eval_regex_match(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        col = self._resolve_column(rule.parameters, role_map, df)
        if not col:
            return self._skip_result(rule, "Column not found")
        pattern = rule.parameters.get("pattern", "")
        if not pattern:
            return self._skip_result(rule, "No pattern specified")

        non_null = df[col].dropna()
        total = len(non_null)
        if total == 0:
            return RuleEvaluationResult(
                rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
                records_checked=0, pass_count=0, fail_count=0,
                score=1.0, target_columns=[col],
            )
        matches = non_null.astype(str).str.match(pattern, na=False).sum()
        pass_count = int(matches)
        fail_count = total - pass_count
        score = pass_count / total
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[col],
        )

    def _eval_allowed_values(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        col = self._resolve_column(rule.parameters, role_map, df)
        if not col:
            return self._skip_result(rule, "Column not found")
        allowed = set(str(v) for v in rule.parameters.get("values", []))
        non_null = df[col].dropna()
        total = len(non_null)
        if total == 0:
            return RuleEvaluationResult(
                rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
                records_checked=0, pass_count=0, fail_count=0,
                score=1.0, target_columns=[col],
            )
        matches = non_null.astype(str).isin(allowed).sum()
        pass_count = int(matches)
        fail_count = total - pass_count
        score = pass_count / total
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[col],
        )

    def _eval_numeric_range(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        col = self._resolve_column(rule.parameters, role_map, df)
        if not col:
            return self._skip_result(rule, "Column not found")
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = numeric.dropna()
        total = len(non_null)
        if total == 0:
            return RuleEvaluationResult(
                rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
                records_checked=0, pass_count=0, fail_count=0,
                score=1.0, target_columns=[col],
            )
        min_val = rule.parameters.get("min_value")
        max_val = rule.parameters.get("max_value")
        inclusive_min = rule.parameters.get("inclusive_min", True)
        inclusive_max = rule.parameters.get("inclusive_max", True)

        mask = pd.Series(True, index=non_null.index)
        if min_val is not None:
            if inclusive_min:
                mask &= non_null >= min_val
            else:
                mask &= non_null > min_val
        if max_val is not None:
            if inclusive_max:
                mask &= non_null <= max_val
            else:
                mask &= non_null < max_val

        pass_count = int(mask.sum())
        fail_count = total - pass_count
        score = pass_count / total
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[col],
        )

    def _eval_column_comparison(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        left, right = self._resolve_two_columns(rule.parameters, role_map, df, "left", "right")
        if not left or not right:
            return self._skip_result(rule, "One or both columns not found")

        operator = rule.parameters.get("operator", ">=")
        left_vals = pd.to_numeric(df[left], errors="coerce")
        right_vals = pd.to_numeric(df[right], errors="coerce")

        # Try datetime if numeric fails
        if left_vals.isna().sum() > len(df) * 0.5:
            left_vals = pd.to_datetime(df[left], errors="coerce")
            right_vals = pd.to_datetime(df[right], errors="coerce")

        both_valid = left_vals.notna() & right_vals.notna()
        total = int(both_valid.sum())
        if total == 0:
            return RuleEvaluationResult(
                rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
                records_checked=0, pass_count=0, fail_count=0,
                score=1.0, target_columns=[left, right],
            )

        lv = left_vals[both_valid]
        rv = right_vals[both_valid]

        if operator == ">=":
            mask = lv >= rv
        elif operator == ">":
            mask = lv > rv
        elif operator == "<=":
            mask = lv <= rv
        elif operator == "<":
            mask = lv < rv
        elif operator == "==":
            mask = lv == rv
        elif operator == "!=":
            mask = lv != rv
        else:
            return self._skip_result(rule, f"Unsupported operator: {operator}")

        pass_count = int(mask.sum())
        fail_count = total - pass_count
        score = pass_count / total
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[left, right],
        )

    def _eval_conditional_required(self, df: pd.DataFrame, rule: RuleDefinition, role_map: dict) -> RuleEvaluationResult:
        cond_col = rule.parameters.get("condition_column")
        cond_val = rule.parameters.get("condition_value")
        req_col = rule.parameters.get("required_column")

        if not cond_col or cond_col not in df.columns:
            return self._skip_result(rule, "Condition column not found")
        if not req_col or req_col not in df.columns:
            return self._skip_result(rule, "Required column not found")

        matching_rows = df[df[cond_col].astype(str) == str(cond_val)]
        total = len(matching_rows)
        if total == 0:
            return RuleEvaluationResult(
                rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
                records_checked=0, pass_count=0, fail_count=0,
                score=1.0, target_columns=[cond_col, req_col],
            )
        fail_count = int(matching_rows[req_col].isna().sum())
        pass_count = total - fail_count
        score = pass_count / total
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=total, pass_count=pass_count, fail_count=fail_count,
            score=score, target_columns=[cond_col, req_col],
        )

    def _skip_result(self, rule: RuleDefinition, error: str) -> RuleEvaluationResult:
        return RuleEvaluationResult(
            rule_key=rule.rule_key, rule_type=rule.rule_type, source=rule.source,
            records_checked=0, pass_count=0, fail_count=0,
            score=0.0, target_columns=[], error=error,
        )
