"""tests/test_ml_persistence.py — Tests for the persisted-model predict paths
added this session, and for the AST-safe rule-condition evaluator.

Builds DataFrame fixtures directly from the dataset CSV via pandas (not
through SQLTool) so these tests don't depend on DATASET_PATH or on module
import order relative to other test files sharing the same pytest process.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.ml.persistence import load_pickle
from app.services.ml.anomaly_detector import AnomalyDetector, RATIO_FEATURE_COLUMNS
from app.services.ml.classifier import RiskSegmenter, VarianceClassifier, SEGMENT_FEATURE_COLS, VARIANCE_FEATURE_COLS
from app.services.ml.forecaster import LightGBMForecaster
from app.services.tools.rule_engine import _safe_eval

_DATASET = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"

pytestmark = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}"
)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_DATASET)


# ── ml/persistence.py ────────────────────────────────────────────────────────

def test_load_pickle_hit():
    obj = load_pickle("isolation_forest.pkl")
    assert obj is not None


def test_load_pickle_miss():
    assert load_pickle("does_not_exist.pkl") is None


# ── AnomalyDetector.score() ──────────────────────────────────────────────────

def test_anomaly_score_missing_column_falls_back_to_none(df):
    incomplete = df[[c for c in RATIO_FEATURE_COLUMNS if c != "premium_collection_rate_actual"]]
    assert AnomalyDetector().score(incomplete) is None


def test_anomaly_score_persisted(df):
    cols = [c for c in RATIO_FEATURE_COLUMNS if c in df.columns]
    result = AnomalyDetector().score(df[cols])
    assert result is not None
    assert result.get("source") == "persisted"
    assert result["anomaly_count"] >= 0


def test_anomaly_score_deterministic_across_calls(df):
    cols = [c for c in RATIO_FEATURE_COLUMNS if c in df.columns]
    subset = df[cols]
    r1 = AnomalyDetector().score(subset)
    r2 = AnomalyDetector().score(subset)
    assert r1["anomaly_count"] == r2["anomaly_count"]
    assert r1["anomalies"] == r2["anomalies"]


# ── RiskSegmenter.assign() ───────────────────────────────────────────────────

def test_segmenter_assign_missing_column_falls_back_to_none(df):
    incomplete = df[[c for c in SEGMENT_FEATURE_COLS if c != "combined_ratio_actual"]]
    assert RiskSegmenter().assign(incomplete) is None


def test_segmenter_assign_persisted(df):
    cols = [c for c in SEGMENT_FEATURE_COLS if c in df.columns]
    result = RiskSegmenter().assign(df[cols])
    assert result is not None
    assert result.get("source") == "persisted"
    assert result["n_clusters"] > 0


def test_segmenter_assign_stable_across_calls(df):
    cols = [c for c in SEGMENT_FEATURE_COLS if c in df.columns]
    subset = df[cols]
    r1 = RiskSegmenter().assign(subset)
    r2 = RiskSegmenter().assign(subset)
    assert r1["segment_distribution"] == r2["segment_distribution"]


# ── VarianceClassifier persisted path ────────────────────────────────────────

def test_classifier_load_persisted():
    clf = VarianceClassifier()
    assert clf.load_persisted() is True
    assert len(clf.classes_) > 0
    assert len(clf.feature_cols) > 0


def test_classifier_predict_batch_missing_column_returns_error(df):
    cols = [c for c in VARIANCE_FEATURE_COLS if c in df.columns and c != "variance_vs_budget_amount"]
    result = VarianceClassifier().predict_batch_with_explanation(df[cols])
    assert result is not None
    assert "error" in result


def test_classifier_predict_batch_with_explanation(df):
    clf = VarianceClassifier()
    clf.load_persisted()
    cols = [c for c in clf.feature_cols if c in df.columns]
    if len(cols) < len(clf.feature_cols):
        pytest.skip("Dataset missing a column the persisted classifier needs")
    result = clf.predict_batch_with_explanation(df[cols].head(100))
    assert result is not None and "error" not in result
    assert result["predicted_driver"] in clf.classes_
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_features"]) > 0


# ── LightGBMForecaster.get_key_drivers() ─────────────────────────────────────

def test_lightgbm_key_drivers_for_trained_target():
    result = LightGBMForecaster().get_key_drivers("underwriting_result_actual", top_n=5)
    assert result is not None
    assert result["source"] == "persisted"
    assert len(result["key_drivers"]) == 5


def test_lightgbm_key_drivers_for_untrained_target_is_none():
    assert LightGBMForecaster().get_key_drivers("some_kpi_never_trained_actual") is None


# ── app/services/rule_engine.py::_safe_eval ──────────────────────────────────
# These 4 conditions use uppercase AND, which is not valid Python — before
# this session's fix, eval() raised SyntaxError on every one of them,
# silently swallowed by evaluate_rules()'s blanket except — so these rules
# had never fired once. Confirm they now evaluate correctly at their
# documented boundary values.

@pytest.mark.parametrize("expr,value,expected", [
    ("75.0 > 75 AND 75.0 <= 100", 80.0, True),   # BR002 loss ratio warning band
    ("95.0 >= 95 AND 95.0 < 100", 97.0, True),   # combined ratio near-breakeven band
    ("1.0 >= 1.0 AND 1.0 < 1.1", 1.05, True),    # reserve adequacy band
])
def test_dormant_and_rules_now_evaluate(expr, value, expected):
    # Rebuild each condition with the actual value substituted in, mirroring
    # what evaluate_rules() does internally.
    templates = {
        "75.0 > 75 AND 75.0 <= 100": f"{value} > 75 AND {value} <= 100",
        "95.0 >= 95 AND 95.0 < 100": f"{value} >= 95 AND {value} < 100",
        "1.0 >= 1.0 AND 1.0 < 1.1": f"{value} >= 1.0 AND {value} < 1.1",
    }
    assert _safe_eval(templates[expr]) == expected


def test_safe_eval_rejects_sandbox_escape_attempts():
    malicious = [
        "__import__('os').system('echo pwned')",
        "().__class__.__base__.__subclasses__()",
        "open('C:/Windows/win.ini').read()",
        "exec('1')",
        "(lambda: 1)()",
        "abs",       # bare identifier, not boolean-producing
        "5",         # bare literal, not boolean-producing
    ]
    for expr in malicious:
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval(expr)


def test_safe_eval_still_allows_abs_in_comparisons():
    assert _safe_eval("-200 < 0 AND abs(-200) > 100") is True
