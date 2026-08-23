"""Parse unified diffs into individually reviewable hunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Literal

HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)

FileStatus = Literal["added", "modified", "deleted", "renamed", "binary"]

# Used when a diff arrives as a bare hunk with no file headers

UNKNOWN_PATH = "<unknown>"

# Machine-generated files produce huge diffs and worthless reviews.
SKIP_SUFFIXES = (
    ".lock",
    ".min.js",
    ".min.css",
    ".map",
    ".snap",
    ".svg",
    ".po",
    ".mo",
)
SKIP_NAMES = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "composer.lock",
    "go.sum",
    "Gemfile.lock",
)


@dataclass(frozen=True)
class Hunk:
    """One ``@@`` block: the exact granularity the adapter was trained on."""

    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[str, ...]

    @property
    def header(self) -> str:
        return (
            f"@@ -{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count} @@{self.section}"
        )

    @property
    def text(self) -> str:
        """Header plus body, formatted exactly like a training-set patch."""
        return "\n".join((self.header, *self.lines))

    @property
    def added_lines(self) -> tuple[tuple[int, str], ...]:
        """``(new_file_line_number, content)`` for each added line."""
        out = []
        line_no = self.new_start
        for line in self.lines:
            if line.startswith("+"):
                out.append((line_no, line[1:]))
                line_no += 1
            elif not line.startswith(("-", "\\")):
                line_no += 1
        return tuple(out)

    @property
    def comment_anchor(self) -> int:
        """New-file line to attach an inline comment to.

        GitHub only accepts inline comments on lines that appear in the diff
        """
        added = self.added_lines
        return added[-1][0] if added else self.new_start


@dataclass(frozen=True)
class FileDiff:
    """All hunks belonging to a single file in the diff."""

    path: str
    old_path: str | None
    status: FileStatus
    hunks: tuple[Hunk, ...]

    @property
    def is_reviewable(self) -> bool:
        if self.status in ("binary", "deleted") or not self.hunks:
            return False
        name = self.path.rsplit("/", 1)[-1]
        return not (name in SKIP_NAMES or self.path.endswith(SKIP_SUFFIXES))


def parse_diff(text: str) -> list[FileDiff]:
    """Split a unified diff into per-file, per-hunk structures."""
    files: list[FileDiff] = []
    for section in _split_file_sections(text):
        parsed = _parse_file_section(section)
        if parsed is not None:
            files.append(parsed)
    return files


def parse_file_patch(
    path: str,
    patch: str,
    status: FileStatus = "modified",
    old_path: str | None = None,
) -> FileDiff:

    return FileDiff(
        path=path,
        old_path=old_path,
        status=status,
        hunks=tuple(_parse_hunks(patch.splitlines(), path)),
    )


def _split_file_sections(text: str) -> Iterator[list[str]]:
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if current:
                yield current
            current = [line]
        elif current or line.startswith(("--- ", "@@ ")):
            # Tolerate bare diffs that have no `diff --git` preamble.
            if not current:
                current = []
            current.append(line)
    if current:
        yield current


def _parse_file_section(lines: list[str]) -> FileDiff | None:
    old_path: str | None = None
    new_path: str | None = None
    status: FileStatus = "modified"
    hunks: list[Hunk] = []

    index = 0
    for index, line in enumerate(lines):
        if line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "deleted"
        elif line.startswith("rename from "):
            status = "renamed"
            old_path = line[len("rename from ") :].strip()
        elif line.startswith("rename to "):
            new_path = line[len("rename to ") :].strip()
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            status = "binary"
        elif line.startswith("--- "):
            old_path = _strip_prefix(line[4:].strip())
        elif line.startswith("+++ "):
            new_path = _strip_prefix(line[4:].strip())
        elif line.startswith("@@"):
            break
    else:
        index = len(lines)

    if new_path is None and old_path is None:
        new_path = _path_from_git_header(lines[0]) if lines else None

    path = new_path or old_path
    if path is None:
        if not lines or not lines[0].startswith("@@"):
            return None
        path = UNKNOWN_PATH

    if status != "binary":
        hunks = _parse_hunks(lines[index:], path)

    return FileDiff(
        path=path,
        old_path=old_path if old_path != path else None,
        status=status,
        hunks=tuple(hunks),
    )


def _parse_hunks(lines: list[str], path: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    pending: dict | None = None
    body: list[str] = []

    def flush() -> None:
        if pending is not None:
            hunks.append(Hunk(**pending, lines=tuple(body)))

    for line in lines:
        match = HUNK_HEADER.match(line)
        if match:
            flush()
            body = []
            old_start, old_count, new_start, new_count, section = match.groups()
            pending = {
                "path": path,
                "old_start": int(old_start),
                # An omitted count means 1 line, per the unified diff spec.
                "old_count": int(old_count) if old_count is not None else 1,
                "new_start": int(new_start),
                "new_count": int(new_count) if new_count is not None else 1,
                "section": section or "",
            }
        elif pending is not None:
            if line.startswith(("+", "-", " ", "\\")) or line == "":
                body.append(line)
            else:
                # Anything else means the file section ended (e.g. trailing junk).
                break

    flush()
    return hunks


def _strip_prefix(path: str) -> str | None:
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _path_from_git_header(line: str) -> str | None:
    match = re.match(r"^diff --git a/(.+) b/(.+)$", line)
    return match.group(2) if match else None


def estimate_tokens(text: str) -> int:
    """Rough token count used when no real tokenizer is supplied.
    """
    return len(text) // 3 + 1


def split_oversized_hunk(
    hunk: Hunk,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[Hunk]:
    """Break one hunk into sequential sub-hunks that each fit the budget.

    """
    if count_tokens(hunk.text) <= max_tokens:
        return [hunk]

    header_cost = count_tokens(hunk.header)
    budget = max_tokens - header_cost
    if budget <= 0:
        raise ValueError(f"max_tokens={max_tokens} too small for a hunk header")

    pieces: list[Hunk] = []
    old_line, new_line = hunk.old_start, hunk.new_start
    batch: list[str] = []
    batch_cost = 0

    def emit() -> None:
        nonlocal old_line, new_line, batch, batch_cost
        if not batch:
            return
        old_count = sum(1 for line in batch if line.startswith(("-", " ")) or line == "")
        new_count = sum(1 for line in batch if line.startswith(("+", " ")) or line == "")
        pieces.append(
            replace(
                hunk,
                old_start=old_line,
                old_count=old_count,
                new_start=new_line,
                new_count=new_count,
                lines=tuple(batch),
            )
        )
        old_line += old_count
        new_line += new_count
        batch = []
        batch_cost = 0

    for line in hunk.lines:
        cost = count_tokens(line)
        if batch and batch_cost + cost > budget:
            emit()
        batch.append(line)
        batch_cost += cost

    emit()
    return pieces


def iter_review_units(
    files: Iterable[FileDiff],
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> Iterator[Hunk]:
    """Yield every reviewable hunk, sized to fit the model's context window."""
    for file_diff in files:
        if not file_diff.is_reviewable:
            continue
        for hunk in file_diff.hunks:
            yield from split_oversized_hunk(hunk, max_tokens, count_tokens)
