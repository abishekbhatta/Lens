"""Prompt text shared by fine-tuning and inference.

The adapter was trained to continue the first-pass prefix with a short review
comment. Inference still uses that prefix so the model answers the question it
was tuned for. A second prompt then rewrites the terse finding into a
clarifying GitHub comment.
"""

from __future__ import annotations

REVIEW_INSTRUCTION = (
    "You are an expert code reviewer. Analyse the following "
    "unified diff and write a concise, actionable review comment."
)

CLARIFY_INSTRUCTION = (
    "You are writing a GitHub pull-request review comment. "
    "The draft below names a finding but is too vague for the author. "
    "Rewrite it in 2-4 sentences that cover all of the following:\n"
    "1. What the diff is changing, naming the identifier or code involved.\n"
    "2. Why that change is a problem, unnecessary, or risky — not merely that it is.\n"
    "3. What the author should do instead, or why the existing code is already enough.\n"
    "The draft conclusion is authoritative: only explain it, do not reverse it. "
    "If the draft says a change is unnecessary, explain why the existing code already covers it. "
    "If the draft says to keep a line, explain where it is still used. "
    "Do not invent a different bug, platform, or install issue. "
    "Do not repeat the diff. Do not wrap the comment in markdown code fences. "
    "Output only the review comment."
)


def build_inference_prompt(diff: str) -> str:
    """Prompt prefix. The model generates the comment after ``[/INST]``."""
    return (
        f"[INST] {REVIEW_INSTRUCTION}\n\n"
        f"```diff\n{diff}\n```\n[/INST]"
    )


def build_clarify_prompt(diff: str, draft: str) -> str:
    """Second-pass prompt: expand a terse finding into a clarifying comment."""
    return (
        f"[INST] {CLARIFY_INSTRUCTION}\n\n"
        f"```diff\n{diff}\n```\n\n"
        f"Draft comment:\n{draft.strip()}\n[/INST]"
    )


def build_training_example(patch: str, msg: str, eos_token: str) -> str:
    """Full supervised example: the inference prefix plus the human comment."""
    return f"{build_inference_prompt(patch)} {msg.strip()}{eos_token}"
