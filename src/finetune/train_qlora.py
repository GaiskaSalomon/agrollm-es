"""Supervised fine-tuning of an open base model with QLoRA.

Stack: HuggingFace `transformers` + `TRL` (SFTTrainer) + `PEFT` (LoRA) +
`bitsandbytes` (4-bit NF4 quantization). Designed to fit a single consumer GPU or a
free Colab T4 by loading the base model in 4-bit and training only LoRA adapters.

Base model defaults to Qwen2.5-0.5B-Instruct (swap for a Llama/Qwen of any size).

Usage (GPU):
    python -m src.finetune.train_qlora \
        --data data/processed/sft.jsonl \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --out outputs/qwen-agro-lora --epochs 3

Then merge / quantize to GGUF for on-prem deployment (see docs/quantization.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QLoRA SFT on an open base model.")
    ap.add_argument("--data", required=True, type=Path, help="SFT JSONL (messages).")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="outputs/qwen-agro-lora", type=Path)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit (use if bitsandbytes is unavailable).")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    # Heavy imports are local so `--help` works without a GPU/torch install.
    # NOTE: import datasets/pyarrow BEFORE torch. On Windows, importing pyarrow after
    # torch triggers a native DLL clash that segfaults the process.
    from datasets import load_dataset
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model (4-bit NF4 by default) ---
    quant_config = None
    if not args.no_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    # --- LoRA adapter config ---
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- Data ---
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    split = dataset.train_test_split(test_size=0.1, seed=13) if len(dataset) >= 10 else None
    train_ds = split["train"] if split else dataset
    eval_ds = split["test"] if split else None

    # --- Trainer (TRL applies the chat template to the "messages" field) ---
    sft_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=torch.cuda.is_available(),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"[train_qlora] LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
