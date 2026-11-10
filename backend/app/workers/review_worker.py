"""Background worker that turns queued pull requests into review comments.

Run it next to the API, on a machine with a GPU:

    python -m app.workers.review_worker

Docker Compose starts this as the ``worker`` service. It loads the model
once, then blocks on Redis until a job arrives. After the model finishes,
the worker publishes the comments as a GitHub review unless
``SENTINEL_POST_REVIEW`` is ``0``.
"""

from __future__ import annotations

import logging
import os

from app.queue.store import JobQueue, get_queue

logger = logging.getLogger(__name__)


def post_reviews_enabled() -> bool:
    raw = os.environ.get("SENTINEL_POST_REVIEW", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def review_pull_request(pr: str, max_comments: int | None) -> dict:
    """Fetch one PR, review its hunks, and publish the comments.

    Model generation is the only GPU step. A GitHub write failure is stored
    on the result and does not discard the comments.
    """
    from dataclasses import asdict

    from app.services.github import GitHubClient, parse_pr_reference
    from app.services.review import review_files

    ref = parse_pr_reference(pr)
    with GitHubClient() as client:
        meta = client.get_pull_request(ref)
        files = client.get_pull_request_files(ref)

    comments = review_files(files, max_comments=max_comments)
    payload = [asdict(comment) for comment in comments]
    result = {
        "title": meta.title,
        "head_sha": meta.head_sha,
        "base_sha": meta.base_sha,
        "comments": payload,
        "posted": False,
        "github_review_id": None,
        "github_review_url": None,
        "post_error": None,
    }
    if not post_reviews_enabled() or not payload:
        return result

    try:
        with GitHubClient() as client:
            posted = client.create_review(ref, meta.head_sha, payload)
    except Exception as exc:
        logger.exception("Failed to post review for %s", ref)
        result["post_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["posted"] = True
    result["github_review_id"] = posted.get("id")
    result["github_review_url"] = posted.get("html_url")
    return result


def process_job(queue: JobQueue, review=review_pull_request, timeout: int = 5) -> bool:
    """Take one job off the queue. Returns False when the wait times out."""
    job_id = queue.dequeue(timeout=timeout)
    if job_id is None:
        return False

    job = queue.get(job_id)
    if job is None:
        logger.warning("Dequeued missing job %s", job_id)
        return True

    logger.info("Reviewing %s (job %s)", job["pr"], job_id)
    queue.mark_running(job_id)
    try:
        result = review(job["pr"], job["max_comments"])
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        queue.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        return True

    queue.mark_succeeded(job_id, result)
    logger.info(
        "Job %s finished with %s comment(s), posted=%s",
        job_id,
        len(result.get("comments", [])),
        result.get("posted"),
    )
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    queue = get_queue()
    queue.ping()
    from app.model.inference import get_reviewer

    logger.info("Loading review model")
    get_reviewer().load()
    logger.info("Worker waiting for review jobs")
    while True:
        process_job(queue)


if __name__ == "__main__":
    main()
