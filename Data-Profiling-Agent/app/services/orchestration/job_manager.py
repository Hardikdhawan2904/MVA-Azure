"""Job manager — abstracts synchronous/async execution behind a unified interface."""

import threading
from datetime import datetime, timezone
from typing import Any, Callable
from enum import Enum

from app.core.enums import RunStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


class JobState:
    """State of a processing job."""

    def __init__(self, run_id: str, status: RunStatus = RunStatus.PENDING):
        self.run_id = run_id
        self.status = status
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.result: Any = None
        self.error: dict[str, Any] | None = None


class ExecutionMode(str, Enum):
    """How jobs are executed."""
    BACKGROUND_THREAD = "background_thread"


class JobManager:
    """
    Manages profiling job lifecycle.

    Runs each job on a background thread, so the public API stays consistent
    even though `submit` returns immediately. When ready for production async:
    swap to Redis/ARQ/Celery without changing the API contracts.
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.BACKGROUND_THREAD):
        self._mode = mode
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create_job(self, run_id: str) -> JobState:
        """Register a new processing job."""
        job = JobState(run_id=run_id, status=RunStatus.PENDING)
        with self._lock:
            self._jobs[run_id] = job
        logger.info("job_created", run_id=run_id, mode=self._mode.value)
        return job

    def submit(self, run_id: str, task: Callable[[], Any]) -> None:
        """Submit a job for execution on a background thread."""
        job = self._jobs.get(run_id)
        if not job:
            return

        thread = threading.Thread(
            target=self._execute,
            args=(run_id, task),
            daemon=True,
            name=f"job-{run_id[:8]}",
        )
        thread.start()

    def _execute(self, run_id: str, task: Callable[[], Any]) -> None:
        """Execute the task and update job state."""
        job = self._jobs.get(run_id)
        if not job:
            return

        with self._lock:
            job.status = RunStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc).isoformat()

        try:
            result = task()
            with self._lock:
                job.status = RunStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.result = result
            logger.info("job_completed", run_id=run_id)
        except Exception as e:
            with self._lock:
                job.status = RunStatus.FAILED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.error = {"code": getattr(e, "code", "PROCESSING_ERROR"), "message": str(e)}
            logger.error("job_failed", run_id=run_id, error=str(e))

    def wait_for_completion(self, run_id: str, timeout: float = 120.0) -> JobState | None:
        """Wait for a job to complete (for API response in sync-like mode)."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self._jobs.get(run_id)
            if not job:
                return None
            if job.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                return job
            time.sleep(0.1)
        return self._jobs.get(run_id)
