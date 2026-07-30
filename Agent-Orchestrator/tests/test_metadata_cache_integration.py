"""tests/test_metadata_cache_integration.py — real-Postgres integration
tests for app/services/dataset_registry/metadata_cache.py.

Every existing Dataset Registry test mocks metadata_cache out entirely —
the actual SQL (CREATE TABLE/INSERT .. ON CONFLICT/UPDATE/_row_to_master's
JSON-field hydration) had zero coverage against a real database. Runs
against the real shared Postgres instance, matching this monorepo's
established practice (Analytics-Agent's test_memory_persistence.py/
test_ml_persistence.py do the same). Skips cleanly if Postgres isn't
reachable — a real infra dependency, not something to fake.

Each test uses its own random fingerprint so tests can't collide with
each other or with real Dataset Registry data.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import pytest

from app.config import settings
from app.services.dataset_registry import metadata_cache
from app.services.dataset_registry.models import DatasetCopy, MasterDataset


def _postgres_reachable() -> bool:
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST, port=settings.POSTGRES_PORT, dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD, connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Shared Postgres not reachable — start the native instance, see Shared-Postgres/README.md",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    metadata_cache.init_db()


def _fake_master(fingerprint: str, **overrides) -> MasterDataset:
    now = datetime.now(timezone.utc)
    base = dict(
        fingerprint=fingerprint, dataset_id=str(uuid.uuid4()),
        original_filename=f"test_{fingerprint[:8]}.csv", storage_location=f"/fake/{fingerprint}.csv",
        file_extension=".csv", byte_size=100, row_count=None, column_count=None,
        latest_version=1, previous_fingerprint=None, reference_count=0,
        first_uploaded_at=now, last_referenced_at=now,
    )
    base.update(overrides)
    return MasterDataset(**base)


def _new_fingerprint() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex  # 64 hex chars, looks like a real sha256


# ── master_datasets round trip ──────────────────────────────────────────────

def test_insert_and_lookup_master_dataset_round_trips():
    fp = _new_fingerprint()
    master = _fake_master(fp)
    metadata_cache.insert_master_dataset(master)

    found = metadata_cache.lookup_master_dataset(fp)
    assert found is not None
    assert found.fingerprint == fp
    assert found.original_filename == master.original_filename
    assert found.latest_version == 1
    assert found.reference_count == 0
    assert found.has_cached_results is False  # no master_dataset_results row yet


def test_lookup_master_dataset_returns_none_for_unknown_fingerprint():
    assert metadata_cache.lookup_master_dataset(_new_fingerprint()) is None


def test_insert_master_dataset_on_conflict_does_nothing():
    """Race-safety property: a second insert for the same fingerprint must
    be a silent no-op, never an error, never overwriting the first row."""
    fp = _new_fingerprint()
    first = _fake_master(fp, original_filename="first.csv")
    second = _fake_master(fp, original_filename="second.csv")

    metadata_cache.insert_master_dataset(first)
    metadata_cache.insert_master_dataset(second)  # must not raise

    found = metadata_cache.lookup_master_dataset(fp)
    assert found.original_filename == "first.csv"  # first insert wins, second was a no-op


def test_update_master_counts_persists():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    metadata_cache.update_master_counts(fp, row_count=42, column_count=7)

    found = metadata_cache.lookup_master_dataset(fp)
    assert found.row_count == 42
    assert found.column_count == 7


def test_touch_last_referenced_updates_timestamp():
    fp = _new_fingerprint()
    master = _fake_master(fp)
    metadata_cache.insert_master_dataset(master)
    before = metadata_cache.lookup_master_dataset(fp).last_referenced_at

    metadata_cache.touch_last_referenced(fp)

    after = metadata_cache.lookup_master_dataset(fp).last_referenced_at
    assert after >= before


def test_lookup_latest_version_by_filename_finds_the_highest_version():
    filename = f"lineage_{uuid.uuid4().hex[:8]}.csv"
    fp_v1 = _new_fingerprint()
    fp_v2 = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp_v1, original_filename=filename, latest_version=1))
    metadata_cache.insert_master_dataset(
        _fake_master(fp_v2, original_filename=filename, latest_version=2, previous_fingerprint=fp_v1),
    )

    found = metadata_cache.lookup_latest_version_by_filename(filename)
    assert found.fingerprint == fp_v2
    assert found.latest_version == 2


# ── master_dataset_results — the actual JSON hydration logic ───────────────

def test_upsert_and_get_cached_result_round_trips_json_fields():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    agent1_body = {"business_domain": "Finance", "quality_report": {"dataset_score": 100}}
    agent2_full_result = {"column_profiles": [{"name": "x"}], "hierarchy": {"status": "unresolved"}}

    metadata_cache.upsert_result(fp, "Finance", agent1_body, agent2_full_result)

    cached = metadata_cache.get_cached_result(fp)
    assert cached is not None
    assert cached.primary_domain == "Finance"
    assert cached.agent1_body == agent1_body
    assert cached.agent2_full_result == agent2_full_result
    assert cached.response_schema_version == metadata_cache.RESPONSE_SCHEMA_VERSION


def test_upsert_result_refreshes_on_conflict():
    """force_revalidate's real use case: a second upsert for the same
    fingerprint must replace the cached result, not be rejected."""
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    metadata_cache.upsert_result(fp, "Finance", {"v": 1}, {"v": 1})
    metadata_cache.upsert_result(fp, "HR", {"v": 2}, {"v": 2})

    cached = metadata_cache.get_cached_result(fp)
    assert cached.primary_domain == "HR"
    assert cached.agent1_body == {"v": 2}


def test_get_cached_result_returns_none_without_a_results_row():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    assert metadata_cache.get_cached_result(fp) is None


def test_lookup_master_dataset_hydrates_metadata_from_real_agent_response_shapes():
    """_row_to_master's JSON extraction, exercised against realistic
    agent1_body/agent2_full_result shapes (not just arbitrary dicts) — this
    is the exact logic the review flagged as having zero real-DB coverage."""
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    agent1_body = {"business_domain": "Finance", "quality_report": {"dataset_score": 99.5, "decision": "PASS"}}
    agent2_full_result = {
        "column_profiles": [{"physical_name": "revenue_actual"}],
        "hierarchy": {"status": "unresolved"},
        "charts": [{"chart_key": "revenue_by_region"}],
        "ml_readiness_breakdown": {"score": 44.2},
        "llm_readiness_breakdown": {"score": 98.85},
        "feature_recommendation": {"target_column": None},
    }
    metadata_cache.upsert_result(fp, "Finance", agent1_body, agent2_full_result)

    found = metadata_cache.lookup_master_dataset(fp)
    assert found.has_cached_results is True
    assert found.validation == {"dataset_score": 99.5, "decision": "PASS"}
    assert found.column_profiles == [{"physical_name": "revenue_actual"}]
    assert found.hierarchy == {"status": "unresolved"}
    assert found.charts == [{"chart_key": "revenue_by_region"}]
    assert found.ml_readiness == {"score": 44.2}
    assert found.llm_readiness == {"score": 98.85}
    assert found.feature_recommendation == {"target_column": None}


# ── dataset_copies ───────────────────────────────────────────────────────────

def test_insert_copy_and_increment_reference_does_both_atomically():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    copy = DatasetCopy(
        copy_id=str(uuid.uuid4()), fingerprint=fp, uploaded_filename="a.csv",
        upload_timestamp=datetime.now(timezone.utc), version=1,
    )

    metadata_cache.insert_copy_and_increment_reference(copy)

    copies = metadata_cache.list_copies_for_fingerprint(fp)
    assert len(copies) == 1
    assert copies[0].copy_id == copy.copy_id
    master = metadata_cache.lookup_master_dataset(fp)
    assert master.reference_count == 1


def test_list_copies_excludes_deleted_unless_requested():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    copy = DatasetCopy(
        copy_id=str(uuid.uuid4()), fingerprint=fp, uploaded_filename="a.csv",
        upload_timestamp=datetime.now(timezone.utc), version=1,
    )
    metadata_cache.insert_copy(copy)

    conn = metadata_cache.get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE dataset_copies SET deleted_flag = TRUE WHERE copy_id = %s", (copy.copy_id,))
    conn.commit()
    cur.close()
    conn.close()

    assert metadata_cache.list_copies_for_fingerprint(fp) == []
    assert len(metadata_cache.list_copies_for_fingerprint(fp, include_deleted=True)) == 1


# ── cascade delete ────────────────────────────────────────────────────────

def test_delete_master_dataset_cascades_to_copies_and_results():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    metadata_cache.upsert_result(fp, "Finance", {}, {})
    copy = DatasetCopy(
        copy_id=str(uuid.uuid4()), fingerprint=fp, uploaded_filename="a.csv",
        upload_timestamp=datetime.now(timezone.utc), version=1,
    )
    metadata_cache.insert_copy(copy)

    metadata_cache.delete_master_dataset(fp)

    assert metadata_cache.lookup_master_dataset(fp) is None
    assert metadata_cache.get_cached_result(fp) is None
    assert metadata_cache.list_copies_for_fingerprint(fp, include_deleted=True) == []


def test_list_master_datasets_includes_freshly_inserted_row():
    fp = _new_fingerprint()
    metadata_cache.insert_master_dataset(_fake_master(fp))
    all_masters = metadata_cache.list_master_datasets(limit=500)
    assert any(m.fingerprint == fp for m in all_masters)
