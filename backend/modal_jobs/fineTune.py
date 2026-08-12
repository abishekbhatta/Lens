from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATASET_DIR = REPO_ROOT / "Dataset"

DATASET_FILES = ("msg-train.jsonl", "msg-valid.jsonl", "msg-test.jsonl")

BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
DATA_MOUNT = Path("/data")
MODEL_MOUNT = Path("/models")
ADAPTER_DIR = MODEL_MOUNT / "adapters" / "sentinel-codellama"
MODAL_SECRET_NAME = "sentinel"

# ---------------------------------------------------------------------------
# Modal app, image, volumes, secrets
# ---------------------------------------------------------------------------

app = modal.App("sentinel")

dataset_vol = modal.Volume.from_name("dataset-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("model-vol", create_if_missing=True)

# Injected into remote containers as os.environ["HF_TOKEN"]
hf_secret = modal.Secret.from_name(MODAL_SECRET_NAME, required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.23.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "transformers==4.40.0",
        "datasets==2.18.0",
        "peft==0.10.0",
        "bitsandbytes==0.43.0",
        "accelerate==0.29.0",
        "trl==0.8.1",
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
    # Cap training rows for first smoke runs; set None to use the full split.
    "max_train_samples": None,
    "max_eval_samples": 2000,
}


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _upload_dataset(force: bool = False) -> None:

    """Push local Dataset/*.jsonl into the Modal dataset volume."""
    
    missing = [name for name in DATASET_FILES if not (LOCAL_DATASET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files under {LOCAL_DATASET_DIR}: {missing}"
        )

    print(f"Uploading dataset from {LOCAL_DATASET_DIR} → volume 'dataset-vol'")
    with dataset_vol.batch_upload(force=force) as batch:
        for name in DATASET_FILES:
            local = LOCAL_DATASET_DIR / name
            remote = f"/{name}"
            print(f"  {local} → {remote} ({local.stat().st_size / 1e9:.2f} GB)")
            batch.put_file(str(local), remote)
    print("Dataset upload complete.")


# ---------------------------------------------------------------------------
# Remote: download / cache base model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Remote: LoRA fine-tune
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A100",
    timeout=60 * 60 * 4,
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
    from datasets import load_dataset
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
            f"{train_path} missing — run with --action upload (or setup) first"
        )

    print("Loading dataset from Modal Volume...")
    data_files = {"train": str(train_path)}
    if valid_path.exists():
        data_files["validation"] = str(valid_path)

    dataset = load_dataset("json", data_files=data_files)

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

    dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)

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


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(action: str = "setup", force_upload: bool = False):
    """
    action:
      upload         — upload Dataset/*.jsonl to dataset-vol
      download-model — cache CodeLlama-7b-Instruct-hf on model-vol
      setup          — upload + download-model
      train          — run LoRA fine-tune
      all            — setup + train
    """
    action = action.strip().lower()
    print(f"Action: {action}")

    if action in {"upload", "setup", "all"}:
        _upload_dataset(force=force_upload)

    if action in {"download-model", "setup", "all"}:
        path = download_base_model.remote(BASE_MODEL)
        print(f"Model cached at: {path}")

    if action in {"train", "all"}:
        adapter = fine_tune.remote()
        print(f"Adapter saved at: {adapter}")

    if action not in {"upload", "download-model", "setup", "train", "all"}:
        raise SystemExit(
            f"Unknown action '{action}'. "
            "Use: upload | download-model | setup | train | all"
        )
