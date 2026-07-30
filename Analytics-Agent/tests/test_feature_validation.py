"""tests/test_feature_validation.py — Tests for
app/services/ml/feature_validation.py, added to cross-check Agent 3's
hardcoded per-model feature column lists against Agent 2's per-upload
column classification.

validate_feature_columns() now takes an already-parsed list[dict] instead
of a file path — app/routes/analyze.py parses the feature_recommendation
Form field's JSON before it ever reaches this function, so there is no
file I/O (and no "file not found" / "malformed JSON" case) left in this
module for these tests to exercise; JSON-parsing failures are now the
route layer's concern.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from app.services.ml.feature_validation import validate_feature_columns
from app.config import ISO_FOREST_CFG, KMEANS_CFG, XGBOOST_CFG, LGBM_CFG


def _complete_fixture() -> list[dict]:
    """A feature-recommendation covering every column every hardcoded list
    expects, all correctly roled — the "nothing wrong" baseline other tests
    mutate one entry of."""
    cols = []
    for col in ISO_FOREST_CFG["feature_columns"]:
        cols.append({"column": col, "role": "metric"})
    for col in KMEANS_CFG["feature_columns"]:
        if col not in {c["column"] for c in cols}:
            cols.append({"column": col, "role": "metric"})
    for col in XGBOOST_CFG["feature_columns"]:
        if col not in {c["column"] for c in cols}:
            cols.append({"column": col, "role": "metric"})
    for col in LGBM_CFG["categorical_features"]:
        cols.append({"column": col, "role": "dimension"})
    return cols


def test_none_is_a_clean_noop(caplog):
    with caplog.at_level(logging.INFO):
        validate_feature_columns(None)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "skipping column validation" in caplog.text


def test_empty_list_is_a_clean_noop(caplog):
    with caplog.at_level(logging.INFO):
        validate_feature_columns([])
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "skipping column validation" in caplog.text


def test_all_correct_produces_no_warnings(caplog):
    with caplog.at_level(logging.INFO):
        validate_feature_columns(_complete_fixture())
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "all hardcoded model columns match" in caplog.text


def test_missing_column_warns_with_model_and_column_named(caplog):
    fixture = [c for c in _complete_fixture() if c["column"] != "loss_ratio_actual"]
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(fixture)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("loss_ratio_actual" in w and "doesn't have it" in w for w in warnings)
    # Both IsolationForest and KMeans use this column — expect both named.
    assert any("IsolationForest" in w and "loss_ratio_actual" in w for w in warnings)
    assert any("KMeans" in w and "loss_ratio_actual" in w for w in warnings)


def test_wrong_role_warns_with_expected_vs_actual(caplog):
    fixture = _complete_fixture()
    for c in fixture:
        if c["column"] == "loss_ratio_actual":
            c["role"] = "dimension"
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(fixture)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "loss_ratio_actual" in w and "'metric'" in w and "'dimension'" in w
        for w in warnings
    )
