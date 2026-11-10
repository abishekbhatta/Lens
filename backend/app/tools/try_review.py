"""Review a diff locally. No GitHub token or pull request required.

Examples (from ``backend/``, with the venv active):

    # Default sample diff under fixtures/
    python -m app.tools.try_review

    # Parse only — no GPU
    python -m app.tools.try_review --parse-only

    # One hunk from the held-out test set
    python -m app.tools.try_review --dataset 0

    # Custom diff file
    python -m app.tools.try_review --diff-file path/to.patch

    # Same diff through the HTTP API (uvicorn must be running)
    python -m app.tools.try_review --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFF = BACKEND_ROOT / "fixtures" / "sample.diff"
DATASET_TEST = BACKEND_ROOT.parent / "Dataset" / "msg-test.jsonl"


def _wrap_bare_patch(patch: str, path: str = "example.py") -> str:
    """Turn a dataset ``patch`` into a minimal unified diff."""
    if patch.lstrip().startswith("diff --git"):
        return patch
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"{patch.strip()}\n"
    )


def load_diff_file(path: Path) -> str:
    return path.read_text()


def load_dataset_row(index: int) -> tuple[str, str | None]:
    if not DATASET_TEST.exists():
        raise FileNotFoundError(f"Test set not found: {DATASET_TEST}")
    with DATASET_TEST.open() as handle:
        for i, line in enumerate(handle):
            if i == index:
                row = json.loads(line)
                patch = row.get("patch") or ""
                msg = (row.get("msg") or "").strip() or None
                return _wrap_bare_patch(patch, "patched.py"), msg
    raise IndexError(f"Dataset row {index} not found in {DATASET_TEST}")


def parse_only(diff: str) -> None:
    from app.services.diff import iter_review_units, parse_diff
    from app.services.review import diff_token_budget
    from app.model.inference import GenerationSettings

    settings = GenerationSettings()
    budget = diff_token_budget(settings, prompt_overhead=80)
    files = parse_diff(diff)
    hunks = list(iter_review_units(files, max_tokens=budget))
    print(f"{len(files)} file(s), {len(hunks)} reviewable hunk(s)\n")
    for hunk in hunks:
        print(f"  {hunk.path}:{hunk.comment_anchor} ({hunk.comment_side})")
        print(f"    {hunk.header}")


def run_model(diff: str, max_comments: int | None) -> None:
    from app.services.review import review_diff

    print("Loading model...", file=sys.stderr)
    comments = review_diff(diff, max_comments=max_comments)
    if not comments:
        print("No comments (empty model output or no reviewable hunks).")
        return
    print(f"\n{len(comments)} comment(s):\n")
    for comment in comments:
        print(f"--- {comment.path}:{comment.line} ({comment.side}) ---")
        print(comment.body)
        print()


def run_api(base_url: str, diff: str, max_comments: int | None) -> None:
    payload: dict = {"diff": diff}
    if max_comments is not None:
        payload["max_comments"] = max_comments
    response = httpx.post(f"{base_url.rstrip('/')}/review", json=payload, timeout=600.0)
    if response.status_code == 503:
        print(
            "API returned 503 (model not loaded). Start uvicorn on a GPU machine:\n"
            "  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
    response.raise_for_status()
    data = response.json()
    for comment in data.get("comments", []):
        print(f"--- {comment['path']}:{comment['line']} ({comment['side']}) ---")
        print(comment["body"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review a diff without calling GitHub."
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        default=None,
        help=f"Unified diff file (default: {DEFAULT_DIFF.name})",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=None,
        metavar="ROW",
        help="Use patch from Dataset/msg-test.jsonl at this zero-based row.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=3,
        help="Cap model calls (default: 3 for quick tests).",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Print reviewable hunks only; do not load the model.",
    )
    parser.add_argument(
        "--api",
        metavar="URL",
        default=None,
        help="POST the diff to a running Sentinel API instead of loading the model locally.",
    )
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="With --dataset, print the human review comment from the dataset.",
    )
    args = parser.parse_args()

    reference: str | None = None
    if args.dataset is not None:
        diff, reference = load_dataset_row(args.dataset)
    else:
        path = args.diff_file or DEFAULT_DIFF
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        if not path.exists():
            raise SystemExit(f"Diff file not found: {path}")
        diff = load_diff_file(path)

    if args.show_reference and reference:
        print("Human reference comment:\n", reference, "\n", sep="")

    if args.parse_only:
        parse_only(diff)
        return

    if args.api:
        run_api(args.api, diff, args.max_comments)
        return

    run_model(diff, args.max_comments)


if __name__ == "__main__":
    main()
