"""
app/services/ml/feature_validation.py — validate the hardcoded per-model
feature column lists in ml_config.yml against Agent 2's per-upload column
classification.

Agent 3's ML feature lists (which ratio columns feed IsolationForest/KMeans,
which variance columns feed XGBoost, which columns LightGBM treats as
categorical) are hand-picked Insurance-domain expertise and stay hardcoded —
Agent 2's own feature_recommendation is a generic, business-question-driven
classification that doesn't map onto Agent 3's fixed internal targets and is
far coarser than what these models actually want (see decision recorded in
the session that added this file).

What this module adds is a per-request cross-check: does *this specific
upload* actually have the columns those hardcoded lists expect, and does
Agent 2 agree they're numeric/dimension columns rather than something else?
A mismatch here would previously only surface as a silent per-query
fallback or a runtime crash deep in a query handler — this makes it visible
at request time instead. Diagnostic only: never raises, never blocks the
request, never changes what any model does.
"""

import logging

from app.config import ISO_FOREST_CFG, KMEANS_CFG, XGBOOST_CFG, LGBM_CFG

logger = logging.getLogger(__name__)

# (config accessor, list key, model name, expected Agent-2 role)
_CHECKS = [
    (ISO_FOREST_CFG, "feature_columns", "IsolationForest", "metric"),
    (KMEANS_CFG, "feature_columns", "KMeans", "metric"),
    (XGBOOST_CFG, "feature_columns", "XGBoost", "metric"),
    (LGBM_CFG, "categorical_features", "LightGBM", "dimension"),
]


def validate_feature_columns(feature_recommendation: list[dict] | None) -> None:
    """Log a warning for every hardcoded feature column that either doesn't
    exist in this upload's schema (per Agent 2's classification) or isn't
    tagged with the expected role. No-ops silently if no recommendation was
    supplied — this is best-effort diagnostics, not a requirement for
    Agent 3 to run. `feature_recommendation` is already-parsed JSON (the
    route parses the `feature_recommendation` Form field before building
    graph state), not a file path — there is no per-upload file to read
    anymore now that Agent 3 receives the upload directly over HTTP."""
    if not feature_recommendation:
        logger.info("No feature recommendation supplied — skipping column validation.")
        return

    roles = {c.get("column"): c.get("role") for c in feature_recommendation if c.get("column")}
    if not roles:
        logger.info("Feature recommendation was empty — skipping column validation.")
        return

    warned = 0
    for cfg, key, model_name, expected_role in _CHECKS:
        for col in cfg.get(key, []):
            if col not in roles:
                logger.warning(
                    f"[feature-validation] {model_name}'s '{key}' expects column "
                    f"'{col}', but this upload's schema doesn't have it."
                )
                warned += 1
            elif roles[col] != expected_role:
                logger.warning(
                    f"[feature-validation] {model_name}'s '{key}' expects column "
                    f"'{col}' to be a '{expected_role}', but Agent 2 classified it "
                    f"as '{roles[col]}' for this upload."
                )
                warned += 1

    if warned == 0:
        logger.info("Feature validation: all hardcoded model columns match this upload's schema. ✓")
