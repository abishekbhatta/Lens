"""Turn a raw unified diff into a list of review comments.

"""

from __future__ import annotations

from dataclasses import dataclass

from app.model.inference import MAX_SEQ_LENGTH, GenerationSettings, get_reviewer
from app.services.diff import FileDiff, Hunk, iter_review_units, parse_diff


#For gitpush only

@dataclass(frozen=True)
class ReviewComment:
    """One model comment, tied to the exact place in the diff that produced it."""

    path: str
    line: int
    side: str
    body: str
    hunk_header: str


def comment_from_hunk(hunk: Hunk, body: str) -> ReviewComment:
    return ReviewComment(
        path=hunk.path,
        line=hunk.comment_anchor,
        side=hunk.comment_side,
        body=body,
        hunk_header=hunk.header,
    )


def diff_token_budget(settings: GenerationSettings, prompt_overhead: int) -> int:
    """Tokens left for diff content once the answer and scaffolding are reserved."""
    return MAX_SEQ_LENGTH - settings.max_new_tokens - prompt_overhead


def review_diff(
    diff: str,
    settings: GenerationSettings | None = None,
    max_comments: int | None = None,
) -> list[ReviewComment]:

    """Review every reviewable hunk in a diff, one model call per hunk."""
    settings = settings or GenerationSettings()
    reviewer = get_reviewer()
    reviewer.load()

    budget = diff_token_budget(settings, reviewer.prompt_overhead)
    units: list[Hunk] = list(
        iter_review_units(
            parse_diff(diff),
            max_tokens=budget,
            count_tokens=reviewer.count_tokens,
        )
    )
    if max_comments is not None:
        units = units[:max_comments]

    comments: list[ReviewComment] = []
    for hunk in units:
        body = reviewer.review(hunk.text, settings)
        if not body:
            continue
        comments.append(comment_from_hunk(hunk, body))
    return comments


def review_files(
    files: list[FileDiff],
    settings: GenerationSettings | None = None,
    max_comments: int | None = None,
) -> list[ReviewComment]:
    """Review hunks from a GitHub files response (already parsed ``FileDiff``s)."""
    settings = settings or GenerationSettings()
    reviewer = get_reviewer()
    reviewer.load()

    budget = diff_token_budget(settings, reviewer.prompt_overhead)
    units: list[Hunk] = list(
        iter_review_units(
            files,
            max_tokens=budget,
            count_tokens=reviewer.count_tokens,
        )
    )
    if max_comments is not None:
        units = units[:max_comments]

    comments: list[ReviewComment] = []
    for hunk in units:
        body = reviewer.review(hunk.text, settings)
        if not body:
            continue
        comments.append(comment_from_hunk(hunk, body))
    return comments
