"""LoRA fine-tune CodeLlama on the local GPU

Run from repo root or backend/:
    python backend/training/fine_tune.py

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Paths / constants

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "Dataset"
MODELS_DIR = REPO_ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "adapters" / "sentinel-codellama"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints" / "sentinel-codellama"

DATASET_FILES = ("msg-train.jsonl", "msg-valid.jsonl", "msg-test.jsonl")
BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"

TRAINING_CONFIG = {
    "base_model": BASE_MODEL,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "num_epochs": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    "max_train_samples": None,  
    "max_eval_samples": None,  
}


def _load_env() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def ensure_dataset() -> Path:
    """Verify Dataset/*.jsonl exists on this machine."""
    missing = [name for name in DATASET_FILES if not (DATASET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files under {DATASET_DIR}: {missing}"
        )
    print(f"Dataset ready at {DATASET_DIR}")
    return DATASET_DIR


def download_base_model(repo_id: str = BASE_MODEL, force: bool = False) -> Path:
    """Download CodeLlama weights into models/ (skipped if already present)."""
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / repo_id
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"Base model already cached at {dest}")
        return dest

    token = _hf_token()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=token,
    )
    print(f"Base model ready at {dest}")
    return dest


def fine_tune(
    *,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Path:
    """LoRA fine-tune CodeLlama on the local GPU."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import DatasetDict, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This script expects a CUDA device on the "
            "college server (check nvidia-smi / CUDA_VISIBLE_DEVICES)."
        )

    cfg = dict(TRAINING_CONFIG)
    if max_train_samples is not None:
        cfg["max_train_samples"] = max_train_samples
    if max_eval_samples is not None:
        cfg["max_eval_samples"] = max_eval_samples

    token = _hf_token()
    local_model = MODELS_DIR / cfg["base_model"]
    model_id = str(local_model) if local_model.exists() else cfg["base_model"]

    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {props.name} "
        f"({props.total_memory / (1024**3):.1f} GiB)"
    )
    print(f"Loading base model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_path = DATASET_DIR / "msg-train.jsonl"
    valid_path = DATASET_DIR / "msg-valid.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing")

    print("Loading dataset from disk...")
    # Load splits separately: train/valid JSONL schemas differ (valid has
    # extra proj/idx/lang columns).
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    splits: dict = {"train": train_ds}
    if valid_path.exists():
        valid_ds = load_dataset("json", data_files=str(valid_path), split="train")
        splits["validation"] = valid_ds
    dataset = DatasetDict(splits)

    def format_prompt(example: dict) -> dict:
        patch = example.get("patch") or ""
        msg = (example.get("msg") or "").strip()
        text = (
            "[INST] You are an expert code reviewer. Analyse the following "
            "unified diff and write a concise, actionable review comment.\n\n"
            f"```diff\n{patch}\n```\n[/INST] {msg}{tokenizer.eos_token}"
        )
        return {"text": text}

    dataset = DatasetDict(
        {
            name: split.map(format_prompt, remove_columns=split.column_names)
            for name, split in dataset.items()
        }
    )
    if cfg["max_train_samples"]:
        n = min(cfg["max_train_samples"], len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    if "validation" in dataset and cfg["max_eval_samples"]:
        n = min(cfg["max_eval_samples"], len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(n))

    print(
        f"Train rows: {len(dataset['train'])}; "
        f"Eval rows: {len(dataset['validation']) if 'validation' in dataset else 0}"
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=training_args,
        packing=False,
    )
    tokenizer.padding_side = "right"

    print("Starting training...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter → {ADAPTER_DIR}")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print("Done.")
    return ADAPTER_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeLlama LoRA adapter on the system GPU."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download the base model (use cache or Hub id).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the base model even if cached.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override TRAINING_CONFIG max_train_samples (omit for config default).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Override TRAINING_CONFIG max_eval_samples.",
    )
    args = parser.parse_args()

    _load_env()
    print("Starting fine-tune pipeline on system GPU...")
    ensure_dataset()

    if not args.skip_download:
        path = download_base_model(BASE_MODEL, force=args.force_download)
        print(f"Model cached at: {path}")

    adapter = fine_tune(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    print(f"Adapter saved at: {adapter}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
