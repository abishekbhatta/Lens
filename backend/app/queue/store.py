"""Redis-backed queue of pull-request review jobs.

The API only writes a job and returns its id. A separate worker process
blocks on the list, runs the model, and writes the result back to the same
key. Callers poll that key instead of holding an HTTP connection open for
the whole review.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[2]

QUEUE_KEY = "sentinel:review:queue"
JOB_PREFIX = "sentinel:job:"
# Finished jobs are kept long enough to poll, then Redis drops them.
RESULT_TTL_SECONDS = 60 * 60 * 24

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(self, client: redis.Redis | None = None, url: str | None = None) -> None:
        if client is not None:
            self._redis = client
            return
        load_dotenv(BACKEND_ROOT / ".env")
        resolved = url or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self._redis = redis.Redis.from_url(resolved, decode_responses=True)

    def ping(self) -> bool:
        return bool(self._redis.ping())

    def enqueue(self, pr: str, max_comments: int | None) -> str:
        job_id = uuid.uuid4().hex
        payload = {
            "id": job_id,
            "pr": pr,
            "max_comments": max_comments,
            "status": QUEUED,
            "error": None,
            "result": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._redis.set(JOB_PREFIX + job_id, json.dumps(payload))
        # Left-push + right-pop makes this a FIFO queue.
        self._redis.lpush(QUEUE_KEY, job_id)
        return job_id

    def dequeue(self, timeout: int = 5) -> str | None:
        """Block until a job id is available, or return None on timeout."""
        item = self._redis.brpop(QUEUE_KEY, timeout=timeout)
        if item is None:
            return None
        _key, job_id = item
        return job_id

    def get(self, job_id: str) -> dict | None:
        raw = self._redis.get(JOB_PREFIX + job_id)
        if raw is None:
            return None
        return json.loads(raw)

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status=RUNNING)

    def mark_succeeded(self, job_id: str, result: dict) -> None:
        self._update(job_id, status=SUCCEEDED, result=result, error=None)
        self._redis.expire(JOB_PREFIX + job_id, RESULT_TTL_SECONDS)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status=FAILED, error=error)
        self._redis.expire(JOB_PREFIX + job_id, RESULT_TTL_SECONDS)

    def _update(self, job_id: str, **fields) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update(fields)
        job["updated_at"] = _now()
        self._redis.set(JOB_PREFIX + job_id, json.dumps(job))


_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    """One client per process. Tests replace this with their own instance."""
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
