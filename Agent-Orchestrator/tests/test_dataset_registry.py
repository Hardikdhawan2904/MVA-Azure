"""tests/test_dataset_registry.py — unit tests for the Dataset Registry
(Stage 0A, see the architecture doc's "Dataset Registry & Lifecycle
Management" section).

No live DB — mirrors this repo's existing convention (test_pipeline_nodes.py)
that Orchestrator DB-touching code is unit-tested against mocks, never a
real Postgres instance, in this test suite.
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.services.dataset_registry.dataset_registry import DatasetRegistry
from app.services.dataset_registry.fingerprint import compute_fingerprint
from app.services.dataset_registry.models import CachedPipelineResult, DatasetCopy, MasterDataset
from app.services.dataset_registry.storage import master_file_path, write_master_file
from app.services.dataset_registry import reference_counter


# ── fingerprint.py ────────────────────────────────────────────────────────

def test_fingerprint_deterministic_for_identical_bytes():
    content = b"a,b,c\n1,2,3\n"
    assert compute_fingerprint(content) == compute_fingerprint(content)


def test_fingerprint_differs_on_one_byte_change():
    original = b"a,b,c\n1,2,3\n"
    edited = b"a,b,c\n1,2,4\n"
    assert compute_fingerprint(original) != compute_fingerprint(edited)


# ── storage.py ────────────────────────────────────────────────────────────

def test_master_file_path_shape():
    path = master_file_path(r"C:\MVAData\master-datasets", "abcd1234", ".csv")
    assert path.name == "abcd1234.csv"
    assert path.parent.name == "ab"


def test_write_master_file_is_atomic_via_replace(tmp_path):
    content = b"a,b\n1,2\n"
    fingerprint = compute_fingerprint(content)
    with patch("app.services.dataset_registry.storage.os.replace") as mock_replace:
        write_master_file(str(tmp_path), fingerprint, ".csv", content)
        assert mock_replace.called
        tmp_arg, final_arg = mock_replace.call_args[0]
        assert str(tmp_arg).endswith(f".{fingerprint}.tmp")
        assert str(final_arg).endswith(f"{fingerprint}.csv")


def test_write_master_file_round_trips_content(tmp_path):
    content = b"a,b\n1,2\n"
    fingerprint = compute_fingerprint(content)
    path = write_master_file(str(tmp_path), fingerprint, ".csv", content)
    assert Path(path).read_bytes() == content


# ── DatasetRegistry.register() ─────────────────────────────────────────────

def _master(fingerprint="fp-existing", dataset_id="ds-1", filename="a.csv", version=1,
            previous_fingerprint=None, reference_count=0):
    now = datetime.now(timezone.utc)
    return MasterDataset(
        fingerprint=fingerprint, dataset_id=dataset_id, original_filename=filename,
        storage_location=f"/store/{fingerprint}.csv", file_extension=".csv", byte_size=100,
        row_count=10, column_count=3, latest_version=version, previous_fingerprint=previous_fingerprint,
        reference_count=reference_count, first_uploaded_at=now, last_referenced_at=now,
    )


def _registry(tmp_path) -> DatasetRegistry:
    return DatasetRegistry(str(tmp_path))


def test_register_genuinely_new_dataset(tmp_path):
    registry = _registry(tmp_path)
    with patch("app.services.dataset_registry.dataset_registry.duplicate_detector.find_duplicate", return_value=None), \
         patch("app.services.dataset_registry.dataset_registry.version_manager.find_prior_version", return_value=None), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_master_dataset") as mock_insert, \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_copy_and_increment_reference") as mock_insert_copy:
        result = registry.register(filename="new.csv", file_extension=".csv", content=b"x,y\n1,2\n")

    assert result.was_duplicate is False
    assert result.is_new_version is False
    assert result.master.latest_version == 1
    assert result.master.previous_fingerprint is None
    assert result.cached_result is None
    mock_insert.assert_called_once()
    # insert_copy_and_increment_reference does both the copy insert AND the
    # reference_count increment atomically, in one call — this is the
    # actual fix under test, so assert it's used instead of two separate
    # non-atomic calls.
    mock_insert_copy.assert_called_once()


def test_register_duplicate_content_reuses_cached_result(tmp_path):
    registry = _registry(tmp_path)
    existing = _master(fingerprint=compute_fingerprint(b"same,bytes\n"), version=3)
    cached = CachedPipelineResult(primary_domain="Finance", agent1_body={"a": 1}, agent2_full_result={"b": 2})

    with patch("app.services.dataset_registry.dataset_registry.duplicate_detector.find_duplicate", return_value=existing), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_copy_and_increment_reference") as mock_insert_copy, \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.touch_last_referenced") as mock_touch, \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.get_cached_result", return_value=cached):
        result = registry.register(filename="same.csv", file_extension=".csv", content=b"same,bytes\n")

    assert result.was_duplicate is True
    assert result.cached_result is cached
    assert result.copy.version == 3
    mock_insert_copy.assert_called_once()
    mock_touch.assert_called_once()


def test_register_duplicate_without_cached_result_yet(tmp_path):
    """Master row exists but master_dataset_results doesn't (e.g. a crash
    between the two writes on an earlier run) — still reported as a
    duplicate (the content really has been seen), but cached_result is
    None so the caller (resolve_dataset_identity) must treat this as a
    cache miss and run the pipeline fresh."""
    registry = _registry(tmp_path)
    existing = _master(fingerprint=compute_fingerprint(b"partial\n"))

    with patch("app.services.dataset_registry.dataset_registry.duplicate_detector.find_duplicate", return_value=existing), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_copy_and_increment_reference"), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.touch_last_referenced"), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.get_cached_result", return_value=None):
        result = registry.register(filename="partial.csv", file_extension=".csv", content=b"partial\n")

    assert result.was_duplicate is True
    assert result.cached_result is None


def test_register_new_version_of_known_filename(tmp_path):
    registry = _registry(tmp_path)
    prior = _master(fingerprint="fp-v1", dataset_id="lineage-1", filename="edits.csv", version=1)

    with patch("app.services.dataset_registry.dataset_registry.duplicate_detector.find_duplicate", return_value=None), \
         patch("app.services.dataset_registry.dataset_registry.version_manager.find_prior_version", return_value=prior), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_master_dataset"), \
         patch("app.services.dataset_registry.dataset_registry.metadata_cache.insert_copy_and_increment_reference"):
        result = registry.register(filename="edits.csv", file_extension=".csv", content=b"edited,content\n1,2\n")

    assert result.was_duplicate is False
    assert result.is_new_version is True
    assert result.master.latest_version == 2
    assert result.master.dataset_id == "lineage-1"
    assert result.master.previous_fingerprint == "fp-v1"


# ── reference_counter.py — delete guardrail ─────────────────────────────────

def test_can_delete_refuses_with_active_copies_unless_forced():
    with patch("app.services.dataset_registry.reference_counter.active_copy_count", return_value=2):
        allowed, reason = reference_counter.can_delete_master("fp", force=False)
        assert allowed is False
        assert "2 active copy" in reason

        allowed, reason = reference_counter.can_delete_master("fp", force=True)
        assert allowed is True
        assert reason is None


def test_can_delete_allows_when_no_active_copies_regardless_of_force():
    with patch("app.services.dataset_registry.reference_counter.active_copy_count", return_value=0):
        allowed, reason = reference_counter.can_delete_master("fp", force=False)
        assert allowed is True
        assert reason is None


def test_delete_master_removes_row_and_file_when_allowed():
    row = {"fingerprint": "fp", "storage_location": "/store/fp.csv"}
    with patch("app.services.dataset_registry.reference_counter.active_copy_count", return_value=0), \
         patch("app.services.dataset_registry.metadata_cache.get_master_dataset_row", return_value=row), \
         patch("app.services.dataset_registry.metadata_cache.delete_master_dataset") as mock_delete, \
         patch("app.services.dataset_registry.storage.delete_master_file") as mock_delete_file:
        ok, reason = reference_counter.delete_master("fp", force=False)

    assert ok is True
    assert reason is None
    mock_delete.assert_called_once_with("fp")
    mock_delete_file.assert_called_once_with("/store/fp.csv")


def test_delete_master_refuses_without_touching_storage(tmp_path):
    with patch("app.services.dataset_registry.reference_counter.active_copy_count", return_value=1), \
         patch("app.services.dataset_registry.metadata_cache.delete_master_dataset") as mock_delete, \
         patch("app.services.dataset_registry.storage.delete_master_file") as mock_delete_file:
        ok, reason = reference_counter.delete_master("fp", force=False)

    assert ok is False
    assert reason is not None
    mock_delete.assert_not_called()
    mock_delete_file.assert_not_called()


# ── resolve_dataset_identity / persist_master_dataset graph nodes ──────────

from app.agents.orchestration_agent.nodes.pipeline import OrchestratorGraphNodes
from app.services.dataset_registry.models import RegisterResult


def _register_result(was_duplicate, cached_result, fingerprint="fp", copy_id="copy-1", version=1):
    master = _master(fingerprint=fingerprint, version=version)
    copy = DatasetCopy(copy_id=copy_id, fingerprint=fingerprint, uploaded_filename="a.csv",
                        upload_timestamp=datetime.now(timezone.utc), version=version)
    return RegisterResult(master=master, copy=copy, was_duplicate=was_duplicate,
                           is_new_version=False, cached_result=cached_result)


def test_resolve_dataset_identity_cache_hit_skips_agent1_agent2():
    nodes = OrchestratorGraphNodes()
    cached = CachedPipelineResult(primary_domain="Finance", agent1_body={"a": 1}, agent2_full_result={"b": 2})
    fake_registry = MagicMock()
    fake_registry.register.return_value = _register_result(was_duplicate=True, cached_result=cached)

    state = {"filename": "a.csv", "content": b"data", "force_reclassify": False, "force_revalidate": False}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = asyncio.run(nodes.resolve_dataset_identity(state))

    assert out["was_cached"] is True
    assert out["agent1_body"] == {"a": 1}
    assert out["agent2_full_result"] == {"b": 2}
    assert out["primary_domain"] == "Finance"
    assert nodes.route_after_identity({**state, **out}) == "cache_hit"


def test_resolve_dataset_identity_cache_miss_for_new_content():
    nodes = OrchestratorGraphNodes()
    fake_registry = MagicMock()
    fake_registry.register.return_value = _register_result(was_duplicate=False, cached_result=None)

    state = {"filename": "a.csv", "content": b"data", "force_reclassify": False, "force_revalidate": False}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = asyncio.run(nodes.resolve_dataset_identity(state))

    assert out["was_cached"] is False
    assert "agent1_body" not in out
    assert nodes.route_after_identity({**state, **out}) == "cache_miss"


def test_resolve_dataset_identity_force_reclassify_bypasses_cache_hit():
    """The critical fix caught during design review: force_reclassify
    (the existing flag) must also force a miss, not just force_revalidate
    — otherwise a caller explicitly asking Agent 1 to redo classification
    would silently get served a cached response that never reaches
    Agent 1 at all."""
    nodes = OrchestratorGraphNodes()
    cached = CachedPipelineResult(primary_domain="Finance", agent1_body={"a": 1}, agent2_full_result={"b": 2})
    fake_registry = MagicMock()
    fake_registry.register.return_value = _register_result(was_duplicate=True, cached_result=cached)

    state = {"filename": "a.csv", "content": b"data", "force_reclassify": True, "force_revalidate": False}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = asyncio.run(nodes.resolve_dataset_identity(state))

    assert out["was_cached"] is False
    assert out["force_reclassify"] is True


def test_resolve_dataset_identity_force_revalidate_implies_force_reclassify():
    nodes = OrchestratorGraphNodes()
    cached = CachedPipelineResult(primary_domain="Finance", agent1_body={"a": 1}, agent2_full_result={"b": 2})
    fake_registry = MagicMock()
    fake_registry.register.return_value = _register_result(was_duplicate=True, cached_result=cached)

    state = {"filename": "a.csv", "content": b"data", "force_reclassify": False, "force_revalidate": True}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = asyncio.run(nodes.resolve_dataset_identity(state))

    assert out["was_cached"] is False
    assert out["force_reclassify"] is True


def test_resolve_dataset_identity_degrades_to_miss_on_registry_error():
    nodes = OrchestratorGraphNodes()
    fake_registry = MagicMock()
    fake_registry.register.side_effect = RuntimeError("Postgres is down")

    state = {"filename": "a.csv", "content": b"data", "force_reclassify": False, "force_revalidate": False}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = asyncio.run(nodes.resolve_dataset_identity(state))  # must not raise

    assert out["was_cached"] is False
    assert out["fingerprint"] is None
    assert out["copy_id"] is None


def test_persist_master_dataset_calls_registry_with_state_fields():
    nodes = OrchestratorGraphNodes()
    fake_registry = MagicMock()
    state = {
        "fingerprint": "fp-123",
        "primary_domain": "Finance",
        "agent1_body": {"row_count": 10, "column_count": 3},
        "agent2_full_result": {"charts": []},
    }
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = nodes.persist_master_dataset(state)

    assert out == {}
    fake_registry.persist_result.assert_called_once_with(
        fingerprint="fp-123", primary_domain="Finance",
        agent1_body={"row_count": 10, "column_count": 3}, agent2_full_result={"charts": []},
        row_count=10, column_count=3,
    )


def test_persist_master_dataset_noop_without_fingerprint():
    nodes = OrchestratorGraphNodes()
    fake_registry = MagicMock()
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = nodes.persist_master_dataset({"fingerprint": None})

    assert out == {}
    fake_registry.persist_result.assert_not_called()


def test_persist_master_dataset_swallows_registry_failure():
    nodes = OrchestratorGraphNodes()
    fake_registry = MagicMock()
    fake_registry.persist_result.side_effect = RuntimeError("Postgres is down")
    state = {"fingerprint": "fp-123", "primary_domain": "Finance", "agent1_body": {}, "agent2_full_result": {}}
    with patch("app.agents.orchestration_agent.nodes.pipeline.get_dataset_registry", return_value=fake_registry):
        out = nodes.persist_master_dataset(state)  # must not raise

    assert out == {}


def test_finalize_includes_dataset_registry_transparency_fields():
    nodes = OrchestratorGraphNodes()
    state = {
        "agent1_body": {"x": 1}, "agent2_full_result": {"y": 2}, "agent3_body": None,
        "primary_domain": "Finance", "fingerprint": "fp-1", "copy_id": "copy-1", "was_cached": True,
    }
    out = nodes.finalize(state)
    assert out["result"]["fingerprint"] == "fp-1"
    assert out["result"]["copy_id"] == "copy-1"
    assert out["result"]["was_cached"] is True


def test_finalize_defaults_was_cached_false_when_absent():
    nodes = OrchestratorGraphNodes()
    state = {"agent1_body": {}, "agent2_full_result": {}, "agent3_body": None, "primary_domain": "Finance"}
    out = nodes.finalize(state)
    assert out["result"]["was_cached"] is False
    assert out["result"]["fingerprint"] is None
