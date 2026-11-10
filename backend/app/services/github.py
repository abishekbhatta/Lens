"""Read pull requests from the GitHub REST API and publish inline reviews.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import httpx
from dotenv import load_dotenv

from app.services.diff import FileDiff, FileStatus, parse_file_patch

BACKEND_ROOT = Path(__file__).resolve().parents[2]

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
# GitHub caps the files endpoint at 100 per page and 3000 files per PR.
FILES_PER_PAGE = 100
MAX_FILE_PAGES = 30

PR_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)/pull/(\d+)"
)
PR_SHORTHAND = re.compile(r"^([^/\s]+)/([^/#\s]+)#(\d+)$")

# GitHub's per-file status vocabulary differs slightly from ours.
STATUS_MAP: dict[str, FileStatus] = {
    "added": "added",
    "removed": "deleted",
    "modified": "modified",
    "renamed": "renamed",
    "copied": "modified",
    "changed": "modified",
    "unchanged": "modified",
}


class GitHubError(RuntimeError):
    """Any failure talking to the GitHub API."""


class PullRequestNotFound(GitHubError):
    pass


class RateLimited(GitHubError):
    pass


def _github_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    return str(body.get("message") or body.get("documentation_url") or response.text[:200])


@dataclass(frozen=True)
class PullRequestRef:
    """Identifies a PR without carrying any of its content."""

    owner: str
    repo: str
    number: int

    def __str__(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"


@dataclass(frozen=True)
class PullRequest:
    ref: PullRequestRef
    title: str
    author: str
    state: str
    draft: bool
    # Needed to post inline review comments: GitHub anchors them to a commit.
    head_sha: str
    base_sha: str
    additions: int
    deletions: int
    changed_files: int


def parse_pr_reference(value: str) -> PullRequestRef:
    """Accept a PR URL or ``owner/repo#number`` shorthand."""
    text = value.strip()
    for pattern in (PR_URL, PR_SHORTHAND):
        match = pattern.match(text)
        if match:
            owner, repo, number = match.groups()
            return PullRequestRef(owner=owner, repo=repo, number=int(number))
    raise ValueError(
        f"Unrecognised pull request reference: {value!r}. Expected "
        "'https://github.com/owner/repo/pull/123' or 'owner/repo#123'."
    )


def github_token() -> str | None:
    load_dotenv(BACKEND_ROOT / ".env")
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def review_summary(count: int) -> str:
    noun = "comment" if count == 1 else "comments"
    return f"Sentinel left {count} inline {noun} on this pull request."


def build_review_payload(
    commit_id: str,
    comments: list[dict],
    body: str | None = None,
) -> dict:
    """JSON body for ``POST /pulls/{n}/reviews``.

    Comments that share a path, line, and side are merged. GitHub rejects a
    review that places two comments on the same line.
    """
    if not commit_id:
        raise ValueError("commit_id is required to post a review")

    merged: dict[tuple[str, int, str], str] = {}
    order: list[tuple[str, int, str]] = []
    for comment in comments:
        path = (comment.get("path") or "").strip()
        text = (comment.get("body") or "").strip()
        try:
            line = int(comment.get("line") or 0)
        except (TypeError, ValueError):
            continue
        side = comment.get("side") or "RIGHT"
        if side not in ("LEFT", "RIGHT"):
            side = "RIGHT"
        if not path or not text or line < 1:
            continue
        key = (path, line, side)
        if key not in merged:
            order.append(key)
            merged[key] = text
        elif text not in merged[key]:
            merged[key] = f"{merged[key]}\n\n{text}"

    if not order:
        raise ValueError("no comments to post")

    return {
        "commit_id": commit_id,
        "body": body if body is not None else review_summary(len(order)),
        "event": "COMMENT",
        "comments": [
            {
                "path": path,
                "body": merged[(path, line, side)],
                "line": line,
                "side": side,
                "subject_type": "line",
            }
            for path, line, side in order
        ],
    }


class GitHubClient:
    """Thin wrapper over the endpoints Sentinel needs."""

    def __init__(
        self,
        token: str | None = None,
        api_root: str = API_ROOT,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_root = api_root.rstrip("/")
        self._owns_client = client is None
        if client is not None:
            self._client = client
            return
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "sentinel-review-bot",
        }
        resolved = token if token is not None else github_token()
        if resolved:
            headers["Authorization"] = f"Bearer {resolved}"
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get_pull_request(self, ref: PullRequestRef) -> PullRequest:
        path = f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}"
        payload = self._get_json(path)
        return PullRequest(
            ref=ref,
            title=payload.get("title") or "",
            author=(payload.get("user") or {}).get("login", ""),
            state=payload.get("state") or "",
            draft=bool(payload.get("draft")),
            head_sha=(payload.get("head") or {}).get("sha", ""),
            base_sha=(payload.get("base") or {}).get("sha", ""),
            additions=payload.get("additions", 0),
            deletions=payload.get("deletions", 0),
            changed_files=payload.get("changed_files", 0),
        )

    def get_pull_request_files(self, ref: PullRequestRef) -> list[FileDiff]:
        """Fetch every changed file, already parsed into reviewable hunks."""
        files: list[FileDiff] = []
        for page in range(1, MAX_FILE_PAGES + 1):
            batch = self._get_json(
                f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/files",
                params={"per_page": FILES_PER_PAGE, "page": page},
            )
            if not batch:
                break

            for entry in batch:
                files.append(_to_file_diff(entry))

            if len(batch) < FILES_PER_PAGE:
                break
        return files

    def get_pull_request_diff(self, ref: PullRequestRef) -> str:
        """Raw unified diff. Raises ``GitHubError`` when the PR is too large."""
        response = self._client.get(
            f"{self.api_root}/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        if response.status_code == 406:
            raise GitHubError(
                f"{ref} is too large for a single diff; use get_pull_request_files()"
            )
        self._raise_for_status(response, ref)
        return response.text

    def create_review(
        self,
        ref: PullRequestRef,
        commit_id: str,
        comments: list[dict],
        body: str | None = None,
    ) -> dict:
        """Submit one pull-request review containing the inline comments.

        ``event`` is COMMENT, so the review is published and does not approve
        the pull request or request changes.
        """
        payload = build_review_payload(commit_id, comments, body)
        response = self._client.post(
            f"{self.api_root}/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews",
            json=payload,
        )
        self._raise_for_status(response, f"{ref} review")
        return response.json()

    def _get_json(self, path: str, params: dict | None = None):
        url = f"{self.api_root}{path}"
        response = self._client.get(url, params=params)
        self._raise_for_status(response, path)
        return response.json()

    def _raise_for_status(self, response: httpx.Response, context) -> None:
        if response.is_success:
            return

        if response.status_code == 404:
            detail = _github_error_message(response)
            hint = (
                "GitHub returned 404 for this pull request number. The PR may not "
                "exist (deleted or never opened), or the repo name is wrong. "
                "For a public read-only smoke test use e.g. "
                "pallets/flask#6176. For this project use "
                "abishekbhatta/Sentinel_#N after you open pull request N on GitHub."
            )
            raise PullRequestNotFound(
                f"{context} not found ({detail}). {hint}"
            )

        # GitHub reports both rate limiting as 403. They provide 5000 requests/ hr
        remaining = response.headers.get("x-ratelimit-remaining")
        if response.status_code in (403, 429) and remaining == "0":
            raise RateLimited(
                "GitHub rate limit exhausted; resets at epoch "
                f"{response.headers.get('x-ratelimit-reset', 'unknown')}"
            )
        if response.status_code == 401:
            raise GitHubError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
        if response.status_code == 403:
            raise GitHubError(
                "GitHub refused the request (403). Writing a review needs a "
                "token with pull request write access. "
                f"{response.text[:200]}"
            )

        raise GitHubError(
            f"GitHub returned {response.status_code} for {context}: "
            f"{response.text[:200]}"
        )


def _to_file_diff(entry: dict) -> FileDiff:
    path = entry.get("filename") or ""
    patch = entry.get("patch")
    status = STATUS_MAP.get(entry.get("status", "modified"), "modified")

    # Binary files above GitHub's per-file size cap arrive with no
    # patch at all

    if not patch:
        return FileDiff(path=path, old_path=None, status="binary", hunks=())

    return parse_file_patch(
        path=path,
        patch=patch,
        status=status,
        old_path=entry.get("previous_filename"),
    )


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Inspect or review a pull request.")
    parser.add_argument(
        "--pr",
        default=None,
        help="PR URL or owner/repo#number (not needed with --diff-file).",
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Review a local unified diff instead of fetching a pull request.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw unified diff instead of a hunk summary.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Run the fine-tuned model on each reviewable hunk and print comments.",
    )
    parser.add_argument(
        "--max-hunks",
        type=int,
        default=None,
        metavar="N",
        help="Cap model calls (default: all reviewable hunks). Use on large PRs.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Publish the generated comments as a GitHub review on the pull request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --post, print the review JSON instead of calling GitHub.",
    )
    args = parser.parse_args()
    if args.post and not args.review:
        parser.error("--post requires --review")
    if args.dry_run and not args.post:
        parser.error("--dry-run requires --post")
    if args.diff_file and args.pr:
        parser.error("Use either --pr or --diff-file, not both")
    if not args.diff_file and not args.pr:
        parser.error("Provide --pr or --diff-file (or use: python -m app.tools.try_review)")

    if args.diff_file:
        if not args.review:
            parser.error("--diff-file requires --review")
        diff_path = args.diff_file
        if not diff_path.is_absolute():
            diff_path = BACKEND_ROOT / diff_path
        if not diff_path.exists():
            raise SystemExit(f"Diff file not found: {diff_path}")
        from app.services.review import review_diff

        print("\nLoading model...", file=sys.stderr)
        comments = review_diff(diff_path.read_text(), max_comments=args.max_hunks)
        if not comments:
            print("\nNo comments generated.")
            return
        for comment in comments:
            print(f"--- {comment.path}:{comment.line} ({comment.side}) ---")
            print(comment.body)
            print()
        if args.post or args.dry_run:
            parser.error("--post/--dry-run need a real --pr (use try_review for local diffs)")
        return

    ref = parse_pr_reference(args.pr)
    try:
        return _run_pr_command(args, ref)
    except PullRequestNotFound as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None


def _run_pr_command(args, ref: PullRequestRef) -> None:
    with GitHubClient() as client:
        if args.raw:
            print(client.get_pull_request_diff(ref))
            return

        pr = client.get_pull_request(ref)
        print(f"{pr.ref}  {pr.title}")
        print(
            f"  by {pr.author} | {pr.state}{' (draft)' if pr.draft else ''} | "
            f"+{pr.additions}/-{pr.deletions} across {pr.changed_files} files"
        )
        print(f"  head {pr.head_sha[:12]}  base {pr.base_sha[:12]}")

        files = client.get_pull_request_files(ref)
        reviewable = [f for f in files if f.is_reviewable]
        print(f"\n{len(files)} files fetched, {len(reviewable)} reviewable\n")
        for file_diff in files:
            mark = "*" if file_diff.is_reviewable else " "
            print(
                f" {mark} {file_diff.path}  [{file_diff.status}] "
                f"{len(file_diff.hunks)} hunks"
            )

    if not args.review:
        return

    from dataclasses import asdict

    from app.services.review import review_files

    print("\nLoading model...", file=sys.stderr)
    comments = review_files(files, max_comments=args.max_hunks)
    if not comments:
        print("\nNo comments generated (no reviewable hunks or empty model output).")
        return

    print(f"\n{len(comments)} comment(s):\n")
    for comment in comments:
        print(f"--- {comment.path}:{comment.line} ({comment.side}) ---")
        print(comment.body)
        print()

    if not args.post:
        return

    payload = build_review_payload(
        pr.head_sha,
        [asdict(comment) for comment in comments],
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    with GitHubClient() as client:
        posted = client.create_review(
            ref,
            pr.head_sha,
            [asdict(comment) for comment in comments],
        )
    url = posted.get("html_url") or posted.get("id")
    print(f"Posted review: {url}")
    return None


if __name__ == "__main__":
    main()
