"""Allows us to make inference using the fine-tuned CodeLlama

Loads the 4-bit base model and the LoRA adapter and turns a unified diff into a commment

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
MODELS_DIR = BACKEND_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "codellama" / "CodeLlama-7b-Instruct-hf"
ADAPTER_DIR = MODELS_DIR / "adapters" / "sentinel-codellama"


INSTRUCTION = (
    "You are an expert code reviewer. Analyse the following"
    "unified diff and write a concise, actionable review comment."
)


MAX_SEQ_LENGTH = 2048 #Context window the adapter was trained with


def build_prompt(diff: str) -> str:
    """Wrap a unified diff in the exact prompt format used during training."""
    return (
        f"[INST] {INSTRUCTION}\n\n"
        f"```diff\n{diff}\n```\n[/INST]"
    )


@dataclass
class GenerationSettings:
    """Decoding knobs. Low temperature keeps reviews grounded in the diff."""

    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


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

    def review(self, diff: str, settings: GenerationSettings | None = None) -> str:
        """Generate a single review comment for one unified diff."""
        if not diff.strip():
            raise ValueError("diff is empty")

        self.load()
        settings = settings or GenerationSettings()

        import torch

        diff = self._fit_to_context(diff, settings.max_new_tokens)
        prompt = build_prompt(diff)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.temperature > 0,
                temperature=settings.temperature,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Slice off the prompt so only the model's answer is decoded.
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _fit_to_context(self, diff: str, max_new_tokens: int) -> str:
        """Trim the diff body so prompt + answer stay inside the context window.

        Truncating the diff rather than the tokenised prompt keeps the closing
        fence and the ``[/INST]`` marker intact; without them the model has no
        cue that it should start answering.
        """
        budget = MAX_SEQ_LENGTH - max_new_tokens - self._prompt_overhead
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    diff = args.diff_file.read_text() if args.diff_file else sys.stdin.read()

    reviewer = get_reviewer()
    print("Loading model...", file=sys.stderr)
    comment = reviewer.review(
        diff,
        GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
    )
    print(comment)


if __name__ == "__main__":
    main()
