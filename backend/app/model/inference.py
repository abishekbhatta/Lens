"""Allows us to make inference using the fine-tuned CodeLlama

Loads the 4-bit base model and the LoRA adapter and turns a unified diff into a commment

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.model.prompts import build_clarify_prompt
from app.model.prompts import build_inference_prompt as build_prompt

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
MODELS_DIR = BACKEND_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "codellama" / "CodeLlama-7b-Instruct-hf"
ADAPTER_DIR = MODELS_DIR / "adapters" / "sentinel-codellama"

MAX_SEQ_LENGTH = 2048  # Context window the adapter was trained with.

# 4-bit 7B fits in ~5 GiB; transformers' allocator warmup peaks around 3.5 GiB extra.
_MIN_GPU_FREE_BYTES = 6 * 1024**3


def _gpu_free_bytes(device: int) -> int:
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return free


def _resolve_gpu_index(requested: int, *, allow_fallback: bool) -> int:
    """Use ``requested`` when it has headroom; otherwise pick the emptiest GPU."""
    import sys

    import torch

    count = torch.cuda.device_count()
    if requested < 0 or requested >= count:
        raise RuntimeError(
            f"Invalid GPU index {requested}; this machine has {count} CUDA device(s)."
        )

    if _gpu_free_bytes(requested) >= _MIN_GPU_FREE_BYTES:
        return requested

    if not allow_fallback:
        free_gib = _gpu_free_bytes(requested) / (1024**3)
        raise RuntimeError(
            f"GPU {requested} has only {free_gib:.2f} GiB free "
            f"(need about {_MIN_GPU_FREE_BYTES / (1024**3):.0f} GiB for 4-bit load). "
            "Free the GPU, pick another with SENTINEL_GPU, or unset SENTINEL_GPU to auto-select."
        )

    best = max(range(count), key=_gpu_free_bytes)
    if _gpu_free_bytes(best) < _MIN_GPU_FREE_BYTES:
        usage = ", ".join(
            f"GPU {i}: {_gpu_free_bytes(i) / (1024**3):.2f} GiB free"
            for i in range(count)
        )
        raise RuntimeError(
            "No CUDA device has enough free memory to load the model. "
            f"Per-GPU free memory: {usage}"
        )

    if best != requested:
        print(
            f"GPU {requested} is low on memory; loading on GPU {best} instead.",
            file=sys.stderr,
        )
    return best


def needs_clarification(comment: str, *, min_words: int = 24) -> bool:
    """True when a first-pass comment is too short to explain the finding."""
    text = comment.strip()
    if not text:
        return False
    return len(text.split()) < min_words


_COMMENT_LABEL = re.compile(
    r"(?im)^(?:rewritten\s+)?(?:review\s+)?comment\s*:\s*"
)


def _clean_comment(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    label = _COMMENT_LABEL.search(text)
    if label:
        text = text[label.end() :].strip()
    elif text.startswith(("@@", "diff --git", "--- ", "+++ ")):
        paragraphs = re.split(r"\n\s*\n", text)
        for part in reversed(paragraphs):
            stripped = part.strip()
            if stripped and not stripped.startswith(("@@", "diff --git", "--- ", "+++ ", "```")):
                text = stripped
                break
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    text = re.sub(r"^```[^\n]*$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pick_comment(draft: str, clarified: str) -> str:
    draft = _clean_comment(draft)
    clarified = _clean_comment(clarified)
    # Keep the adapter finding when the rewrite is empty or still a one-liner.
    if not clarified or needs_clarification(clarified):
        return draft
    return clarified


@dataclass
class GenerationSettings:
    """Decoding knobs. Low temperature keeps reviews grounded in the diff."""

    # First-pass outputs follow the training set (often a short finding). Extra
    # tokens here come out of the diff budget, so this stays modest.
    max_new_tokens: int = 160
    min_new_tokens: int = 0
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    # Rewrite terse adapter output into a 2-4 sentence GitHub comment.
    clarify: bool = True
    clarify_max_new_tokens: int = 256
    clarify_min_new_tokens: int = 72


class DiffReviewer:
    """Holds the loaded model so a process pays the ~10s load cost only once."""

    def __init__(
        self,
        base_model_dir: Path = BASE_MODEL_DIR,
        adapter_dir: Path = ADAPTER_DIR,
        gpu_index: int | None = None,
    ) -> None:
        self.base_model_dir = base_model_dir
        self.adapter_dir = adapter_dir
        self._gpu_explicit = gpu_index is not None or "SENTINEL_GPU" in os.environ
        self.gpu_index = gpu_index if gpu_index is not None else int(
            os.environ.get("SENTINEL_GPU", "0")
        )
        self._model = None
        self._tokenizer = None
        self._prompt_overhead = 0

    def load(self) -> None:
        """Load base weights in 4-bit, then apply the LoRA adapter on top."""

        if self._model is not None:
            return

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not available. 4-bit inference requires a CUDA device "
                "(check nvidia-smi / CUDA_VISIBLE_DEVICES)."
            )
        if not self.adapter_dir.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {self.adapter_dir}. "
                "Run: python backend/training/fine_tune.py"
            )

        self.gpu_index = _resolve_gpu_index(
            self.gpu_index, allow_fallback=not self._gpu_explicit
        )
        torch.cuda.set_device(self.gpu_index)

        # Same quantisation as training, so the adapter sees the weights it was tuned against.

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_dir),
            quantization_config=quantization_config,
            # Pin to one GPU: a 4-bit 7B fits in ~5 GiB, so sharding only adds latency.
            device_map={"": self.gpu_index},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        model.eval()
        model.config.use_cache = True

        # The adapter directory carries the tokenizer saved at the end of training,
        # including the pad-token alignment that run applied.
        tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self._model = model
        self._tokenizer = tokenizer
        self._prompt_overhead = len(tokenizer(build_prompt(""))["input_ids"])

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def prompt_overhead(self) -> int:
        """Tokens consumed by the prompt scaffolding alone, excluding the diff."""
        self.load()
        return self._prompt_overhead

    def count_tokens(self, text: str) -> int:
        """Exact token count, so chunking can use the model's own tokenizer."""
        self.load()
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def review(self, diff: str, settings: GenerationSettings | None = None) -> str:
        """Generate a single review comment for one unified diff."""
        if not diff.strip():
            raise ValueError("diff is empty")

        self.load()
        settings = settings or GenerationSettings()

        fitted = self._fit_to_context(diff, settings.max_new_tokens)
        draft = self._generate(build_prompt(fitted), settings)
        if not draft:
            return ""
        if not settings.clarify or not needs_clarification(draft):
            return _clean_comment(draft)
        return self._clarify(diff, draft, settings)

    def _clarify(self, diff: str, draft: str, settings: GenerationSettings) -> str:
        """Rewrite a terse first-pass finding into a clarifying comment."""
        clarify_settings = GenerationSettings(
            max_new_tokens=settings.clarify_max_new_tokens,
            min_new_tokens=settings.clarify_min_new_tokens,
            temperature=min(settings.temperature, 0.3),
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            clarify=False,
        )
        overhead = len(self._tokenizer(build_clarify_prompt("", draft))["input_ids"])
        fitted = self._fit_to_context(
            diff, clarify_settings.max_new_tokens, prompt_overhead=overhead
        )
        prompt = build_clarify_prompt(fitted, draft)
        # The LoRA adapter was trained to emit short findings. Disable it so the
        # base Instruct model can follow the rewrite instruction.
        disable = getattr(self._model, "disable_adapter", None)
        if disable is not None:
            with disable():
                clarified = self._generate(prompt, clarify_settings)
        else:
            clarified = self._generate(prompt, clarify_settings)
        return _pick_comment(draft, clarified)

    def _generate(self, prompt: str, settings: GenerationSettings) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generate_kwargs = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "repetition_penalty": settings.repetition_penalty,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if settings.min_new_tokens:
            generate_kwargs["min_new_tokens"] = min(
                settings.min_new_tokens, settings.max_new_tokens
            )
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _fit_to_context(
        self,
        diff: str,
        max_new_tokens: int,
        prompt_overhead: int | None = None,
    ) -> str:
        """Trim the diff body so prompt + answer stay inside the context window.

        Truncating the diff rather than the tokenised prompt keeps the closing
        fence and the ``[/INST]`` marker intact; without them the model has no
        cue that it should start answering.
        """
        overhead = self._prompt_overhead if prompt_overhead is None else prompt_overhead
        budget = MAX_SEQ_LENGTH - max_new_tokens - overhead
        if budget <= 0:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} leaves no room for the diff "
                f"in a {MAX_SEQ_LENGTH}-token context"
            )

        token_ids = self._tokenizer(diff, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= budget:
            return diff
        return self._tokenizer.decode(token_ids[:budget], skip_special_tokens=True)


_reviewer: DiffReviewer | None = None


def get_reviewer() -> DiffReviewer:
    """Process-wide reviewer, so the API and worker load the model only once."""
    global _reviewer
    if _reviewer is None:
        load_dotenv(BACKEND_ROOT / ".env")
        adapter = os.environ.get("SENTINEL_ADAPTER_DIR")
        if adapter:
            _reviewer = DiffReviewer(adapter_dir=Path(adapter))
        else:
            _reviewer = DiffReviewer()
    return _reviewer


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a review comment for a unified diff."
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        help="Path to a unified diff. Reads stdin when omitted.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip the second pass that expands terse findings.",
    )
    args = parser.parse_args()

    diff = args.diff_file.read_text() if args.diff_file else sys.stdin.read()

    reviewer = get_reviewer()
    print("Loading model...", file=sys.stderr)
    comment = reviewer.review(
        diff,
        GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            clarify=not args.no_clarify,
        ),
    )
    print(comment)


if __name__ == "__main__":
    main()
