"""tests/test_database.py — Schema-Intelligence-Layer's Postgres service
layer. Runs against the real shared Postgres instance, matching this
monorepo's established practice (Analytics-Agent's test_memory_persistence.py/
test_ml_persistence.py do the same). Skips cleanly if Postgres isn't
reachable — a real infra dependency, not something to fake.

Covers the dataset-ID race condition found during a handover code review:
get_next_dataset_id() used to do SELECT COUNT(*) then format an ID in a
separate transaction from the INSERT, so two concurrent uploads could read
the same count and collide on the dataset_id PRIMARY KEY. Fixed with a real
Postgres SEQUENCE (nextval()), which is atomic under concurrency by
construction.
"""

import concurrent.futures
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import pytest

from app.services import database
from app.config import settings


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
    database.init_db()


def test_get_next_dataset_id_returns_the_expected_format():
    dataset_id = database.get_next_dataset_id()
    assert dataset_id.startswith("DS_")
    assert dataset_id[3:].isdigit()


def test_get_next_dataset_id_is_unique_under_concurrent_callers():
    """The actual regression test for the race condition: concurrent calls
    must never produce a duplicate ID. Before the fix (SELECT COUNT(*) +
    format, no atomicity), this reliably reproduced collisions under real
    concurrent load. Worker count stays within the connection pool's
    max size (_MAX_POOL_CONNECTIONS=15, database.py) — this test is about
    proving no ID collisions, not about stress-testing pool exhaustion,
    which is a separate, legitimate, expected failure mode of a bounded
    pool under genuinely excessive load."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(lambda _: database.get_next_dataset_id(), range(10)))

    assert len(results) == len(set(results)), f"duplicate IDs generated: {results}"


def test_get_next_dataset_id_never_regresses_below_existing_max():
    """init_db()'s reseed must only ever move the sequence forward — a
    second init_db() call (e.g. a service restart) must not reset it and
    risk re-colliding with rows that already exist."""
    first = database.get_next_dataset_id()
    database.init_db()  # simulates a restart re-running the reseed
    second = database.get_next_dataset_id()

    first_n = int(first.split("_")[1])
    second_n = int(second.split("_")[1])
    assert second_n > first_n
