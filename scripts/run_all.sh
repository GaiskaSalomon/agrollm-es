#!/usr/bin/env bash
# End-to-end pipeline. Steps 1-2 run on CPU; step 3 needs a GPU (Colab T4 works);
# steps 4-5 need the pgvector container (docker compose up -d).
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> 1. Clean"
python -m src.data_pipeline.clean   --in data/raw/corpus.jsonl --out data/interim/clean.jsonl
echo ">> 2. Dedup"
python -m src.data_pipeline.dedup   --in data/interim/clean.jsonl --out data/interim/dedup.jsonl
echo ">> 3. Build SFT dataset"
python -m src.data_pipeline.build_sft_dataset --in data/interim/dedup.jsonl --out data/processed/sft.jsonl

echo ">> 4. Fine-tune (QLoRA) — requires GPU; comment out if running CPU-only"
python -m src.finetune.train_qlora --data data/processed/sft.jsonl --out outputs/qwen-agro-lora || \
  echo "   (skipped fine-tuning — no GPU)"

echo ">> 5. RAG ingest + eval — requires 'docker compose up -d'"
python -m src.rag.ingest --in data/interim/dedup.jsonl
python -m src.eval.evaluate --report outputs/eval_report.json

echo ">> Done."
