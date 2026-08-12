from __future__ import annotations

from pathlib import Path

import modal

# Paths / constants

DATASET_FILES = ("msg-train.jsonl", "msg-valid.jsonl", "msg-test.jsonl")

BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
DATA_MOUNT = Path("/data")
MODEL_MOUNT = Path("/models")
ADAPTER_DIR = MODEL_MOUNT / "adapters" / "sentinel-codellama"
MODAL_SECRET_NAME = "sentinel"


def _local_dataset_dir() -> Path:

    """Dataset/ on the machine running ``modal run`` (not in-container)."""
    return Path(__file__).resolve().parents[2] / "Dataset"



# Modal app, image, and volumes

app = modal.App("sentinel")

dataset_vol = modal.Volume.from_name("dataset-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("model-vol", create_if_missing=True)


hf_secret = modal.Secret.from_name(MODAL_SECRET_NAME, required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.23.0") # For faster concurrent chunked downloding         
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(

        # torch==2.2.0 is built against NumPy 1.x; 
        "numpy<2",
        "torch==2.2.0",
        "transformers==4.40.0",
        "datasets==2.18.0",
        "peft==0.10.0",
        "bitsandbytes==0.43.0",
        "accelerate==0.29.0",
        "trl==0.8.1",
        "rich",  # required by trl.trainer.utils
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
)

TRAINING_CONFIG = {
    "base_model": BASE_MODEL,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "num_epochs": 3,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,


    # Full train set is ~118k rows; set None to use all.

    "max_train_samples": 40_000, #Set to 40k for smaller run since A1000 is limited
    "max_eval_samples": 2000,
}


# Helper functions


def _upload_dataset(force: bool = False) -> None:
    """Push local Dataset/*.jsonl into the Modal dataset volume.

    Skips files that already exist unless ``force`` is True.
    """
    local_dataset_dir = _local_dataset_dir()
    missing = [name for name in DATASET_FILES if not (local_dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files under {local_dataset_dir}: {missing}"
        )

    existing: set[str] = set()
    if not force:
        try:
            existing = {entry.path.lstrip("/") for entry in dataset_vol.listdir("/")}
        except Exception:
            # Empty / brand-new volume — nothing to skip.
            existing = set()

    to_upload = [name for name in DATASET_FILES if force or name not in existing]
    if not to_upload:
        print("Dataset already on volume 'dataset-vol'; skipping upload.")
        return

    print(f"Uploading dataset from {local_dataset_dir} → volume 'dataset-vol'")
    with dataset_vol.batch_upload(force=force) as batch:
        for name in to_upload:
            local = local_dataset_dir / name
            remote = f"/{name}"
            print(f"  {local} → {remote} ({local.stat().st_size / 1e9:.2f} GB)")
            batch.put_file(str(local), remote)
    print("Dataset upload complete.")


# Download / cache base model

@app.function(
    image=download_image,
    volumes={MODEL_MOUNT.as_posix(): model_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def download_base_model(repo_id: str = BASE_MODEL) -> str:

    """Download CodeLlama weights into the model volume (once)."""

    import os

    from huggingface_hub import snapshot_download

    hf_token = os.environ["HF_TOKEN"]
    dest = MODEL_MOUNT / repo_id
    print(f"Downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=hf_token,
    )
    model_vol.commit()
    print(f"Base model ready at {dest}")
    return str(dest)



# Remote: LoRA fine-tune

@app.function(
    image=train_image,
    gpu="A100",
    timeout=60 * 60 * 7,  # headroom for ~5h train + load/eval
    secrets=[hf_secret],
    volumes={
        DATA_MOUNT.as_posix(): dataset_vol,
        MODEL_MOUNT.as_posix(): model_vol,
    },
)
def fine_tune() -> str:
    """LoRA fine-tune CodeLlama on the uploaded PR-review dataset."""
    import os

    import torch
    from datasets import DatasetDict, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    cfg = TRAINING_CONFIG
    hf_token = os.environ["HF_TOKEN"]
    local_model = MODEL_MOUNT / cfg["base_model"]
    model_id = str(local_model) if local_model.exists() else cfg["base_model"]

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
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
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

    train_path = DATA_MOUNT / "msg-train.jsonl"
    valid_path = DATA_MOUNT / "msg-valid.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} missing — dataset not found on volume 'dataset-vol'"
        )

    print("Loading dataset from Modal Volume...")

    # Load splits separately: train/valid JSONL schemas differ (valid has
    # extra proj/idx/lang columns), and a single load_dataset(data_files=...)
    # requires identical columns across files.


    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    splits = {"train": train_ds}
    if valid_path.exists():
        valid_ds = load_dataset("json", data_files=str(valid_path), split="train")
        splits["validation"] = valid_ds
    dataset = DatasetDict(splits)

    def format_prompt(example: dict) -> dict:
        # Match inference: review from the unified diff → free-text comment.
        patch = example.get("patch") or ""
        msg = (example.get("msg") or "").strip()
        text = (
            "[INST] You are an expert code reviewer. Analyse the following "
            "unified diff and write a concise, actionable review comment.\n\n"
            f"```diff\n{patch}\n```\n[/INST] {msg}{tokenizer.eos_token}"
        )
        return {"text": text}

    # Per-split map: column sets differ between train and validation.
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

    out_dir = "/tmp/sentinel-adapter"
    training_args = TrainingArguments(
        output_dir=out_dir,
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

    print("Starting training...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter → {ADAPTER_DIR}")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    model_vol.commit()
    print("Done. Adapter persisted on volume 'model-vol'.")
    return str(ADAPTER_DIR)


# Local entrypoint 

@app.local_entrypoint()
def main(force_upload: bool = False):

    """Upload dataset, cache CodeLlama on model-vol, then run LoRA fine-tune."""
    print("Starting full fine-tune pipeline...")

    _upload_dataset(force=force_upload)

    path = download_base_model.remote(BASE_MODEL)
    print(f"Model cached at: {path}")

    adapter = fine_tune.remote()
    print(f"Adapter saved at: {adapter}")
    print("Pipeline complete.")
