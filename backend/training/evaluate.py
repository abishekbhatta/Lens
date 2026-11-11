"""Score held-out review comments against the human reference.

Formatting and unigram overlap need no GPU. Generation loads the fine-tuned
adapter and is off unless you pass ``--generate``:

    python -m training.evaluate
    python -m training.evaluate --generate 20

The test split is ``Dataset/msg-test.jsonl``. Overlap is token unigram F1
between the model's comment and the human ``msg``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.model.prompts import build_inference_prompt  # noqa: E402

DATASET_PATH = REPO_ROOT / "Dataset" / "msg-test.jsonl"


def unigram_f1(prediction: str, reference: str) -> float:
    """Overlap of whitespace tokens, lowercased. 1.0 is a perfect multiset match."""
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_f1(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(unigram_f1(pred, ref) for pred, ref in pairs) / len(pairs)


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def format_report(rows: list[dict]) -> dict:
    """Confirm every row can be wrapped in the training prompt."""
    missing = 0
    for row in rows:
        patch = row.get("patch") or ""
        if not patch.strip() or not (row.get("msg") or "").strip():
            missing += 1
            continue
        build_inference_prompt(patch)
    return {
        "rows": len(rows),
        "missing_patch_or_msg": missing,
        "ready": len(rows) - missing,
    }


def generate_scores(rows: list[dict]) -> dict:
    from app.model.inference import GenerationSettings, get_reviewer

    reviewer = get_reviewer()
    reviewer.load()
    # Score the first-pass adapter output against the short human labels.
    settings = GenerationSettings(clarify=False)
    pairs: list[tuple[str, str]] = []
    for index, row in enumerate(rows, start=1):
        patch = row.get("patch") or ""
        reference = (row.get("msg") or "").strip()
        if not patch.strip() or not reference:
            continue
        prediction = reviewer.review(patch, settings)
        pairs.append((prediction, reference))
        print(f"[{index}] f1={unigram_f1(prediction, reference):.3f}", file=sys.stderr)
    return {"scored": len(pairs), "mean_unigram_f1": mean_f1(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the held-out review test split.")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        metavar="N",
        help="Load the adapter and score N test comments. 0 skips generation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="JSONL file with patch and msg fields.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    rows = load_rows(args.dataset, args.generate if args.generate > 0 else None)
    report = format_report(rows)
    print(
        f"Formatted {report['ready']} of {report['rows']} rows "
        f"({report['missing_patch_or_msg']} missing patch or msg)."
    )
    if args.generate <= 0:
        print("Pass --generate N to score the model against the human comments.")
        return

    scored = generate_scores(rows[: args.generate])
    print(
        f"Scored {scored['scored']} comments. "
        f"Mean unigram F1: {scored['mean_unigram_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
