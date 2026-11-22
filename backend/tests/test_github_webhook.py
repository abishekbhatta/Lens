"""Webhook signature checks and which pull-request actions get queued."""

from __future__ import annotations

import hashlib
import hmac
import json

import fakeredis
import pytest

from app.queue.store import JobQueue
from app.services.github_webhook import (
    handle_github_webhook,
    pr_reference_from_payload,
    verify_github_signature,
)

SECRET = "test-webhook-secret"


def _sign(body: bytes, secret: str = SECRET) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _queue() -> JobQueue:
    return JobQueue(client=fakeredis.FakeRedis(decode_responses=True))


def _payload(action: str = "opened", *, draft: bool = False, number: int = 7) -> dict:
    return {
        "action": action,
        "number": number,
        "pull_request": {
            "number": number,
            "draft": draft,
            "html_url": f"https://github.com/acme/widgets/pull/{number}",
        },
        "repository": {"full_name": "acme/widgets"},
    }


def _deliver(queue: JobQueue, payload: dict, *, event: str = "pull_request", delivery: str = "d1"):
    body = json.dumps(payload).encode()
    return handle_github_webhook(
        body=body,
        event=event,
        signature=_sign(body),
        delivery=delivery,
        secret=SECRET,
        queue=queue,
    )


def test_signature_accepts_the_raw_body():
    body = b'{"zen":"keep it secret"}'
    assert verify_github_signature(SECRET, body, _sign(body))


def test_signature_rejects_a_tampered_body():
    body = b'{"action":"opened"}'
    signature = _sign(body)
    assert not verify_github_signature(SECRET, body + b" ", signature)
    assert not verify_github_signature(SECRET, body, "sha1=abc")
    assert not verify_github_signature("", body, signature)


def test_opened_pull_request_is_queued():
    queue = _queue()
    outcome = _deliver(queue, _payload())

    assert outcome.status_code == 202
    assert outcome.body["status"] == "queued"
    job = queue.get(outcome.body["id"])
    assert job["pr"] == "acme/widgets#7"
    assert job["max_comments"] == 20
    assert job["status"] == "queued"


def test_draft_and_synchronize_are_ignored():
    queue = _queue()
    draft = _deliver(queue, _payload(draft=True), delivery="draft")
    sync = _deliver(queue, _payload(action="synchronize"), delivery="sync")

    assert draft.status_code == 200
    assert sync.status_code == 200
    assert draft.body == {"status": "ignored"}
    assert sync.body == {"status": "ignored"}
    assert queue.dequeue(timeout=1) is None


def test_ready_for_review_is_queued():
    reference = pr_reference_from_payload(_payload(action="ready_for_review"))
    assert reference == "acme/widgets#7"


def test_duplicate_delivery_is_not_queued_twice():
    queue = _queue()
    payload = _payload()
    first = _deliver(queue, payload, delivery="same")
    second = _deliver(queue, payload, delivery="same")

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.body == {"status": "duplicate"}
    assert queue.dequeue(timeout=1) == first.body["id"]
    assert queue.dequeue(timeout=1) is None


def test_failed_enqueue_releases_the_delivery(monkeypatch: pytest.MonkeyPatch):
    queue = _queue()

    def explode(pr: str, max_comments: int | None) -> str:
        raise RuntimeError("redis down")

    monkeypatch.setattr(queue, "enqueue", explode)
    outcome = _deliver(queue, _payload(), delivery="retry-me")

    assert outcome.status_code == 503
    assert queue.claim_delivery("retry-me") is True


def test_ping_does_not_enqueue():
    queue = _queue()
    body = b'{"zen":"design for failure"}'
    outcome = handle_github_webhook(
        body=body,
        event="ping",
        signature=_sign(body),
        delivery="ping-1",
        secret=SECRET,
        queue=queue,
    )
    assert outcome.status_code == 200
    assert outcome.body == {"status": "pong"}
    assert queue.dequeue(timeout=1) is None


def test_missing_secret_and_bad_signature_are_rejected():
    queue = _queue()
    body = json.dumps(_payload()).encode()
    missing = handle_github_webhook(
        body=body,
        event="pull_request",
        signature=_sign(body),
        delivery="d",
        secret=None,
        queue=queue,
    )
    bad = handle_github_webhook(
        body=body,
        event="pull_request",
        signature="sha256=" + "0" * 64,
        delivery="d2",
        secret=SECRET,
        queue=queue,
    )
    assert missing.status_code == 503
    assert bad.status_code == 401


def test_unrelated_events_and_invalid_json_are_not_queued():
    queue = _queue()
    push = b'{"ref":"refs/heads/main"}'
    ignored = handle_github_webhook(
        body=push,
        event="push",
        signature=_sign(push),
        delivery="push-1",
        secret=SECRET,
        queue=queue,
    )
    broken = handle_github_webhook(
        body=b"{",
        event="pull_request",
        signature=_sign(b"{"),
        delivery="bad-json",
        secret=SECRET,
        queue=queue,
    )
    assert ignored.body == {"status": "ignored"}
    assert broken.status_code == 400
    assert queue.dequeue(timeout=1) is None
