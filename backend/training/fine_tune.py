"""LoRA fine-tune CodeLlama on the local GPU.

Run from repo root:
    python backend/training/fine_tune.py

"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "Dataset"
MODELS_DIR = BACKEND_ROOT / "models"
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


def _ensure_dataset() -> None:
    missing = [name for name in DATASET_FILES if not (DATASET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files under {DATASET_DIR}: {missing}"
        )
    print(f"Dataset ready at {DATASET_DIR}")


def _ensure_base_model() -> Path:
    """Download base weights if not already cached locally."""
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / BASE_MODEL
    if dest.exists() and any(dest.iterdir()):
        print(f"Base model already cached at {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {BASE_MODEL} → {dest}")
    snapshot_download(
        repo_id=BASE_MODEL,
        local_dir=str(dest),
        token=_hf_token(),
    )
    print(f"Base model ready at {dest}")
    return dest


def _fine_tune() -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import DatasetDict, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This script expects a CUDA device "
            "(check nvidia-smi / CUDA_VISIBLE_DEVICES)."
        )

    cfg = TRAINING_CONFIG
    token = _hf_token()
    model_id = str(MODELS_DIR / cfg["base_model"])

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / (1024**3):.1f} GiB)")
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

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    train_path = DATASET_DIR / "msg-train.jsonl"
    valid_path = DATASET_DIR / "msg-valid.jsonl"

    print("Loading dataset from disk...")
    splits: dict = {
        "train": load_dataset("json", data_files=str(train_path), split="train")
    }
    if valid_path.exists():
        splits["validation"] = load_dataset(
            "json", data_files=str(valid_path), split="train"
        )
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

    steps_per_epoch = max(
        1,
        len(dataset["train"])
        // (cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]),
    )
    warmup_steps = max(
        1, int(steps_per_epoch * cfg["num_epochs"] * cfg["warmup_ratio"])
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(CHECKPOINT_DIR),
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["per_device_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if "validation" in dataset else "no",
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,
            report_to="none",
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
            packing=False,
            loss_type="nll",
        ),
    )

    print("Starting training...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter → {ADAPTER_DIR}")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print("Done.")
    return ADAPTER_DIR


def main() -> None:
    _load_env()
    print("Starting fine-tune pipeline...")
    _ensure_dataset()
    _ensure_base_model()
    adapter = _fine_tune()
    print(f"Adapter saved at: {adapter}")


if __name__ == "__main__":
    main()
