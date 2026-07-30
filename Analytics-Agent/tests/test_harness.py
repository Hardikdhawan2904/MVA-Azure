"""
tests/test_harness.py — Automated 12-Case Iterative Test Harness

Executes 12 diverse prompt variations, validates narrative outputs programmatically,
checks for errors/exceptions/tracebacks, and prints a test matrix.
"""

import os
import sys
import logging
from pathlib import Path

import pytest

# Project paths setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.agents.analytics_agent.graph import run_analytics_graph

# Setup simple console logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TestHarness")

_DATASET = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"

pytestmark = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}"
)

# 12 diverse prompts to validate all intents, filters, and fallback paths
TEST_CASES = [
    {
        "id": 1,
        "name": "Show KPI (Amount)",
        "query": "Show Gross Written Premium for EMEA in FY2025",
        "ml_readiness": 99.75,
        "checks": ["gross written premium", "emea", "2025"]
    },
    {
        "id": 2,
        "name": "Show KPI (Ratio)",
        "query": "Show loss ratio for General Insurance in Q3 2025",
        "ml_readiness": 99.75,
        "checks": ["loss ratio", "general insurance", "q3", "2025"]
    },
    {
        "id": 3,
        "name": "Variance vs Budget (Amount)",
        "query": "Show Gross Written Premium variance vs budget for APAC in FY2025",
        "ml_readiness": 99.75,
        "checks": ["gross written premium", "variance", "budget", "apac", "2025"]
    },
    {
        "id": 4,
        "name": "Variance vs Budget (Ratio)",
        "query": "Show combined ratio variance vs budget for General Insurance segment in FY2025",
        "ml_readiness": 99.75,
        "checks": ["combined ratio", "variance", "budget", "general insurance", "2025"]
    },
    {
        "id": 5,
        "name": "YoY Variance (Prior Year)",
        # Uses Gross Written Premium, not Net Written Premium: NWP's curated
        # KPI definition (kpi_definitions.json) has no prior_year_column --
        # the dataset genuinely has no net_written_premium_prior_year column,
        # so a prior-year question about NWP correctly falls back to a
        # budget-only answer with no "prior year" text at all. That's
        # correct system behavior, not something to test around. GWP does
        # have a real prior_year_column, so it actually exercises this path.
        "query": "Compare Gross Written Premium vs prior year for EMEA in Q2 2025",
        "ml_readiness": 99.75,
        "checks": ["gross written premium", "prior year", "emea", "q2", "2025"]
    },
    {
        "id": 6,
        "name": "Root Cause Analysis (Drop)",
        "query": "Why did underwriting result decline in EMEA in FY2025?",
        "ml_readiness": 99.75,
        "checks": ["underwriting result", "emea", "2025", "driver"]
    },
    {
        "id": 7,
        "name": "Root Cause Analysis (Increase)",
        "query": "What drove the underwriting result increase in Q4 2025?",
        "ml_readiness": 99.75,
        "checks": ["underwriting result", "q4", "2025", "driver"]
    },
    {
        "id": 8,
        "name": "Trend Analysis (Historical)",
        "query": "Show the trend of loss ratio for property line of business",
        "ml_readiness": 99.75,
        # The filter always resolves correctly to line_of_business="Property
        # Insurance" (deterministic), but the LLM sometimes narrates it as
        # "property line of business" instead of literally "property
        # insurance" — same fact, different wording. Accept either.
        "checks": ["loss ratio", "trend", ("property insurance", "property line of business")]
    },
    {
        "id": 9,
        "name": "ML Forecast (Prophet)",
        "query": "Forecast underwriting result for next 6 months",
        "ml_readiness": 99.75,
        "checks": ["underwriting result", "forecast", "2026"]
    },
    {
        "id": 10,
        "name": "ML Forecast (Fallback)",
        "query": "Forecast underwriting result for next 6 months",
        "ml_readiness": 50.0,
        # "readiness score" was too literal a phrase to require -- the LLM
        # narrator reliably explains the ML-readiness
        # gate in its own words ("ML readiness being below the required
        # threshold") without using that exact two-word phrase. "readiness"
        # alone is still a genuine signal (the gate was mentioned at all)
        # without pinning down its phrasing, same rationale as case 8's
        # accept-either pattern below.
        #
        # Known flaky (confirmed live, 2/3 runs passed with zero code
        # changes between them): this query can resolve to either a single
        # "forecast" analysis or an unrestricted multi-analysis "report
        # mode" response depending on live intent interpretation, and
        # whether the Azure OpenAI narration call itself succeeds or hits a
        # rate-limit and falls back to the raw template formatter changes
        # whether "readiness" ends up phrased in a way this substring
        # check catches. This is inherent to asserting on live-LLM prose
        # rather than deterministic evidence fields — not a code bug.
        "checks": ["underwriting result", "readiness", "trend"]
    },
    {
        "id": 11,
        "name": "ML Anomaly (Isolation Forest)",
        "query": "Detect anomalies in financial ratios",
        "ml_readiness": 99.75,
        "checks": ["anomaly", "anomalies", "ratio"]
    },
    {
        "id": 12,
        "name": "ML Segmentation (Fallback)",
        "query": "Segment portfolio by risk profile",
        "ml_readiness": 50.0,
        "checks": ["segment", "risk profile", "low risk", "high risk"]
    }
]


def run_test_case(tc: dict) -> tuple[bool, str]:
    """Execute a single query against the graph and validate the response."""
    try:
        result = run_analytics_graph(
            file_content=_DATASET.read_bytes(),
            business_question=tc["query"],
            ml_readiness_score=tc["ml_readiness"],
            llm_readiness_score=99.75,
            # Real callers get this from the Orchestrator (Agent 1's domain
            # classification); a direct call like this one must state its
            # domain assumption explicitly rather than relying on an
            # implicit fallback — see GenericDomainPlugin's docstring
            # (Phase 4) for why silently defaulting to Insurance for any
            # unlabeled dataset was a real bug.
            detected_domain="Insurance",
        )
        if result["status"] != "ok":
            return False, f"Graph reported status={result['status']}: {result['response']}"
        response = result["response"]

        # Check 1: Response must be a string and not empty
        if not response or not isinstance(response, str):
            return False, "Empty or invalid response type"
            
        # Check 2: Must not contain any Python/SQL exceptions (ignore "exceptional" to prevent false positive)
        response_lower = response.lower()
        response_no_exceptional = response_lower.replace("exceptional", "")
        error_keywords = ["traceback", "exception", "attributeerror", "typeerror", "valueerror", "syntaxerror", "db-error", "invalid query", "no such column"]
        for kw in error_keywords:
            if kw in response_no_exceptional:
                return False, f"Detected error signature: '{kw}'"
                
        # Check 3: Structured template format headers must be present (Summary and Supporting Evidence or similar)
        required_headers = ["## summary", "## supporting evidence"]
        for header in required_headers:
            if header not in response_lower:
                return False, f"Missing required markdown section: '{header}'"
                
        # Check 4: Check if specified keywords exist in response (dynamic check).
        # A check entry can be a single required string, or a tuple/list of
        # alternates (at least one must appear) — for facts the LLM narrator
        # might phrase in more than one equally-correct way.
        for kw in tc["checks"]:
            if isinstance(kw, (list, tuple)):
                if not any(alt.lower() in response_lower for alt in kw):
                    return False, f"Missing required evidence keyword (any of): {kw}"
            elif kw.lower() not in response_lower:
                return False, f"Missing required evidence keyword: '{kw}'"
                
        return True, "Passed"
    except Exception as e:
        return False, f"Crashed with exception: {str(e)}"


def execute_suite() -> int:
    """Run all 12 tests and output the matrix. Returns number of failures."""
    print("\n" + "="*80)
    print("  AUTOMATED 12-CASE TEST HARNESS RUN")
    print("="*80)
    print(f"{'ID':<4} | {'Test Name':<30} | {'Status':<10} | {'Details / Notes'}")
    print("-"*80)
    
    failures = 0
    for tc in TEST_CASES:
        passed, msg = run_test_case(tc)
        status_str = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            failures += 1
        print(f"{tc['id']:<4} | {tc['name']:<30} | {status_str:<10} | {msg}")
        
    print("="*80)
    print(f"Total Test Cases: {len(TEST_CASES)}  |  Passed: {len(TEST_CASES) - failures}  |  Failed: {failures}")
    print("="*80 + "\n")
    return failures


# Previously invisible to `pytest tests/` — execute_suite()/run_test_case()
# had no test_-prefixed entry point, so pytest silently collected 0 items
# from this file despite it exercising real end-to-end query paths.
# Parametrized so each of the 12 cases reports individually under pytest,
# same as `python test_harness.py` reports them individually in its matrix.
# Note: each case spins up a fresh AnalyticsAgent (real Azure OpenAI calls) — slower
# and more rate-limit-prone than the rest of the suite.
@pytest.mark.parametrize(
    "tc", TEST_CASES, ids=[f"case_{tc['id']}_{tc['name']}" for tc in TEST_CASES]
)
def test_harness_case(tc):
    passed, msg = run_test_case(tc)
    assert passed, msg


if __name__ == "__main__":
    sys.exit(execute_suite())
