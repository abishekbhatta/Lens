"""Turn a GitHub ``pull_request`` delivery into a queued review.

A new pull request (``opened``) and a draft that becomes ready
(``ready_for_review``) are enqueued. The worker fetches the diff, reviews
it, and posts the comments. Drafts and later pushes are left alone so a
review is not published on every commit.
"""

from __future__ import annotations

# import hashlib
import hmac
import json
import logging
from dataclasses import dataclass

from app.queue.store import JobQueue

logger = logging.getLogger(__name__)

# Matches the default on ``POST /jobs``.
DEFAULT_MAX_COMMENTS = 20
# ``opened`` covers a non-draft pull request. ``ready_for_review`` covers a
# draft once the author marks it ready.
REVIEW_ACTIONS = frozenset({"opened", "ready_for_review"})


@dataclass(frozen=True)
class WebhookOutcome:
    status_code: int
    body: dict


def verify_github_signature(secret: str, body: bytes, signature: str | None) -> bool:
    """Check ``X-Hub-Signature-256`` against the raw request body."""
    if not secret or not signature or not signature.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    expected = f"sha256={digest}"
    if len(expected) != len(signature):
        return False
    return hmac.compare_digest(expected, signature)


def pr_reference_from_payload(payload: dict) -> str | None:
    """Return ``owner/repo#number`` when this delivery should be reviewed."""
    action = payload.get("action")
    if action not in REVIEW_ACTIONS:
        return None

    pull_request = payload.get("pull_request") or {}
    if action == "opened" and pull_request.get("draft"):
        return None

    repository = payload.get("repository") or {}
    full_name = repository.get("full_name") or ""
    number = pull_request.get("number")
    if not full_name or not isinstance(number, int) or number < 1:
        return None
    return f"{full_name}#{number}"


def handle_github_webhook(
    *,
    body: bytes,
    event: str | None,
    signature: str | None,
    delivery: str | None,
    secret: str | None,
    queue: JobQueue,
    max_comments: int | None = DEFAULT_MAX_COMMENTS,
) -> WebhookOutcome:
    """Verify one delivery and enqueue a review when it is a new pull request."""
    if not secret:
        logger.error("GITHUB_WEBHOOK_SECRET is not set; rejecting webhook")
        return WebhookOutcome(503, {"detail": "webhook secret not configured"})

    if not verify_github_signature(secret, body, signature):
        logger.warning("Rejected GitHub webhook with an invalid signature")
        return WebhookOutcome(401, {"detail": "invalid signature"})

    if event == "ping":
        return WebhookOutcome(200, {"status": "pong"})

    if event != "pull_request":
        return WebhookOutcome(200, {"status": "ignored"})

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return WebhookOutcome(400, {"detail": "invalid json"})
    if not isinstance(payload, dict):
        return WebhookOutcome(400, {"detail": "invalid json"})

    pr = pr_reference_from_payload(payload)
    if pr is None:
        logger.info(
            "Ignoring pull_request webhook action=%s delivery=%s",
            payload.get("action"),
            delivery,
        )
        return WebhookOutcome(200, {"status": "ignored"})

    if delivery:
        try:
            claimed = queue.claim_delivery(delivery)
        except Exception:
            logger.exception("Failed to record GitHub delivery %s", delivery)
            return WebhookOutcome(503, {"detail": "job queue unavailable"})
        if not claimed:
            logger.info("Ignoring duplicate delivery %s for %s", delivery, pr)
            return WebhookOutcome(200, {"status": "duplicate"})

    try:
        job_id = queue.enqueue(pr, max_comments)
    except Exception:
        logger.exception("Failed to enqueue review for %s", pr)
        if delivery:
            try:
                queue.release_delivery(delivery)
            except Exception:
                logger.exception("Failed to release delivery %s", delivery)
        return WebhookOutcome(503, {"detail": "job queue unavailable"})

    logger.info("Queued review %s for %s (delivery %s)", job_id, pr, delivery)
    return WebhookOutcome(202, {"status": "queued", "id": job_id})
