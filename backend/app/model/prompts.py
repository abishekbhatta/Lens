"""Prompt text shared by fine-tuning and inference.

The adapter was trained to continue this exact prefix with a short review
comment. Inference has to send the same prefix or the model is answering a
different question than the one it was tuned for.
"""

from __future__ import annotations

REVIEW_INSTRUCTION = (
    "You are an expert code reviewer. Analyse the following "
    "unified diff and write a concise, actionable review comment."
)


def build_inference_prompt(diff: str) -> str:
    """Prompt prefix. The model generates the comment after ``[/INST]``."""
    return (
        f"[INST] {REVIEW_INSTRUCTION}\n\n"
        f"```diff\n{diff}\n```\n[/INST]"
    )


def build_training_example(patch: str, msg: str, eos_token: str) -> str:
    """Full supervised example: the inference prefix plus the human comment."""
    return f"{build_inference_prompt(patch)} {msg.strip()}{eos_token}"
