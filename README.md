# AgroLLM-ES 🌱

**Spanish domain LLM pipeline: custom dataset → QLoRA fine-tuning → RAG on PostgreSQL/pgvector → evaluation.**

A small end-to-end project that adapts an open base model (Qwen2.5) to a Spanish
technical domain: agronomy, irrigation and hydrology. It covers the full workflow —
building the dataset, fine-tuning with QLoRA, grounding answers with RAG, and measuring
quality — and runs locally on a single GPU.

> Author: **Gaiska Salomón** — PhD candidate in Statistics & Data Science.
> The domain corpus comes from my own field (irrigation engineering / hydrology).

---

## Why this project

Generic LLMs are weak on specialized Spanish technical content: they invent crop
coefficients, confuse irrigation methods, and cite non-existent norms. This repo shows
the standard recipe to fix that:

1. **Custom dataset** — collect a Spanish domain corpus, clean it, deduplicate it, and
   turn it into an instruction (SFT) dataset.
2. **Fine-tune** — supervised fine-tuning with **QLoRA** (4-bit) using HuggingFace
   `transformers` + `TRL` + `PEFT`. Runs on a single consumer GPU / free Colab T4.
3. **RAG** — ground answers in the corpus with a retrieval pipeline on
   **PostgreSQL + pgvector**, so factual claims are sourced.
4. **Evaluate** — an evaluation suite with a lightweight **hallucination / groundedness**
   check.

This mirrors the real lifecycle: *data pipeline → CPT/SFT → LoRA/QLoRA → quantization →
RAG → eval*.

---

## Architecture

```
 raw corpus (es)            data_pipeline/                  finetune/
┌──────────────┐   clean   ┌────────────┐  build_sft  ┌──────────────────┐
│ corpus.jsonl │ ────────► │ clean+dedup │ ──────────► │ SFT JSONL        │
└──────────────┘           └────────────┘             │ (messages)       │
                                 │                     └────────┬─────────┘
                                 │                              │ QLoRA / TRL
                                 │ ingest (chunks+embeddings)   ▼
                                 ▼                     ┌──────────────────┐
                          ┌──────────────┐            │ Qwen2.5 + adapter│
                          │ PostgreSQL   │            └────────┬─────────┘
                          │ + pgvector   │◄───── RAG retrieve ─┘
                          └──────────────┘
                                 │
                                 ▼
                          eval/ groundedness + hallucination report
```

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the dataset (CPU, runs anywhere)
python -m src.data_pipeline.clean   --in data/raw/corpus.jsonl --out data/interim/clean.jsonl
python -m src.data_pipeline.dedup   --in data/interim/clean.jsonl --out data/interim/dedup.jsonl
python -m src.data_pipeline.build_sft_dataset --in data/interim/dedup.jsonl --out data/processed/sft.jsonl

# 3. Fine-tune with QLoRA (GPU — use free Colab T4 if no local GPU)
python -m src.finetune.train_qlora --data data/processed/sft.jsonl --out outputs/qwen-agro-lora

# 4. RAG on PostgreSQL + pgvector
docker compose up -d                         # starts postgres+pgvector
python -m src.rag.ingest  --in data/interim/dedup.jsonl
python -m src.rag.query   --q "¿Cómo se calcula la lámina de riego para maíz?"

# 5. Evaluate (+ hallucination check)
python -m src.eval.evaluate --report outputs/eval_report.json
```

No local GPU? Step 3 runs as-is on a free **Google Colab T4** (QLoRA 4-bit fits a 0.5B–3B
base). Steps 1, 2, 4 and 5 run on CPU.

---

## Results

Trained locally on an **NVIDIA RTX 3060 (12 GB)**, base `Qwen/Qwen2.5-0.5B-Instruct`,
QLoRA 4-bit, 3 epochs (~19 s). Evaluated on a held-out domain test set with the RAG
pipeline. Metrics from `outputs/eval_report.json`:

| Metric | Value |
|---|---|
| keyword_recall | **1.00** |
| semantic_sim (vs reference) | **0.78** |
| groundedness (answer vs retrieved context) | **0.67** |
| hallucination_rate | **0.00** |

**Before / after (same question, same retrieval):**

> *¿Cómo se calcula la lámina de riego para maíz?*

- **Base model (no fine-tune, no RAG):** invents a non-existent *"fórmula RRA"* and
  unrelated units — confident hallucination.
- **Fine-tuned + RAG:** uses the correct domain relation `ETc = ET0 × Kc`, the real Kc
  values from the corpus (0.4 initial → 0.7 maturation), and cites retrieved sources.

A 0.5B model is intentionally small so it trains on a laptop GPU; scale the `--base`
flag to a 3B/7B Qwen or Llama for production quality — the pipeline is unchanged.

## Running on Windows (notes)

This was developed and run on Windows 11 + RTX 3060. Two native-DLL ordering quirks and
their fixes (already applied in the code):

- Import `datasets`/`pyarrow` **before** `torch`, and load the embedding model **before**
  opening the `psycopg` connection — importing them in the reverse order triggers an
  OpenMP/`libpq` DLL clash that segfaults the process.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` before training.

On Linux (the production target) these quirks do not occur.

## Maps to the role (Promtec — LLM Engineer)

| What you'll build (job) | Where in this repo |
|---|---|
| Spanish dataset pipeline: cleaning, dedup, tokenization | `src/data_pipeline/` |
| CPT/SFT on Llama/Qwen | `src/finetune/train_qlora.py` (Qwen2.5) |
| SFT with LoRA/QLoRA, HuggingFace + TRL | `train_qlora.py` (TRL `SFTTrainer` + PEFT) |
| Quantization for on-prem (GGUF, llama.cpp) | `docs/quantization.md` |
| RAG on PostgreSQL + pgvector | `src/rag/` + `docker-compose.yml` |
| Evaluation suite + hallucination monitoring | `src/eval/evaluate.py` |

---

## Tech stack

Python · PyTorch · HuggingFace `transformers` / `datasets` · `TRL` · `PEFT` ·
`bitsandbytes` (4-bit) · `sentence-transformers` · PostgreSQL + `pgvector` · Docker.

## Repo layout

```
agrollm-es/
├── data/
│   ├── raw/corpus.jsonl          # seed Spanish domain corpus (runs out of the box)
│   ├── interim/                  # cleaned + dedup output
│   └── processed/                # final SFT dataset
├── src/
│   ├── data_pipeline/            # clean.py, dedup.py, build_sft_dataset.py
│   ├── finetune/train_qlora.py   # TRL SFTTrainer + QLoRA
│   ├── rag/                      # db.py, ingest.py, query.py (pgvector)
│   └── eval/evaluate.py          # eval + hallucination/groundedness
├── docs/quantization.md          # GGUF / llama.cpp / Ollama notes
├── docker-compose.yml            # postgres + pgvector
└── requirements.txt
```

## License

MIT.
