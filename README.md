# AgroLLM-ES 🌱

A small end-to-end project that adapts an open base model (Qwen2.5) to a Spanish
technical domain: agronomy, irrigation and hydrology. It covers the full workflow —
building the dataset, fine-tuning with QLoRA, grounding answers with RAG, and measuring
quality — and runs locally on a single GPU.

> Author: **Gaiska Salomón** — PhD candidate in Statistics & Data Science.
> The domain corpus comes from my own field (irrigation engineering / hydrology).

---

## Why this project

Generic LLMs are weak on specialized Spanish technical content: they invent crop
coefficients, confuse irrigation methods, and cite non-existent norms. This repo implements
the standard recipe and measures how much each step helps:

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
┌──────────────┐   clean   ┌─────────────┐  build_sft  ┌──────────────────┐
│ corpus.jsonl │ ────────► │ clean+dedup │ ──────────► │     SFT JSONL    │
└──────────────┘           └─────────────┘             │    (messages)    │
                                 │                     └────────┬─────────┘
                                 │                              │ QLoRA / TRL
                                 │ ingest (chunks+embeddings)   ▼
                                 ▼                     ┌──────────────────┐
                          ┌──────────────┐             │ Qwen2.5 + adapter│
                          │  PostgreSQL  │             └────────┬─────────┘
                          │ + pgvector   │◄─────  RAG retrieve ─┘
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
python -m src.eval.evaluate --adapter none --report outputs/eval_base_rag.json   # base model + RAG
```

No local GPU? Step 3 runs as-is on a free **Google Colab T4** (QLoRA 4-bit fits a 0.5B–3B
base). Steps 1, 2, 4 and 5 run on CPU.

---

## Results

Base `Qwen/Qwen2.5-0.5B-Instruct`, QLoRA 4-bit, 3 epochs (~43 s on an **NVIDIA RTX 3060, 12 GB**),
trained on 45 examples (43 corpus documents + 2 hand-written worked examples). Both systems answer
through the same RAG pipeline (top-4 chunks from pgvector, greedy decoding, one run).

**Evaluation set** (`src/eval/testset.jsonl`, 100 hand-written questions over a 59-document corpus):

| Group | n | What it tests |
|---|---|---|
| answerable, *seen* | 55 | the gold document was in the fine-tuning data |
| answerable, *unseen* | 25 | the gold document was **withheld from fine-tuning** (`"heldout": true`) but is in the RAG index, so the answer has to come from retrieval |
| unanswerable | 20 | no answer in the corpus; the right behaviour is to say so |

| Group | Metric | Base + RAG | Fine-tuned + RAG |
|---|---|---|---|
| seen (n=55) | keyword recall | 0.79 (95% CI 0.68–0.89) | 0.70 (0.58–0.81) |
| seen (n=55) | all keywords present | 0.76 | 0.65 |
| unseen (n=25) | keyword recall | 0.78 (0.62–0.92) | 0.74 (0.56–0.90) |
| unseen (n=25) | all keywords present | 0.76 | 0.72 |
| answerable (n=80) | retrieval hit (gold doc in top-4) | 0.95 | 0.95 |
| unanswerable (n=20) | abstained correctly | 2 / 20 | 1 / 20 |
| unanswerable (n=20) | hallucination rate (groundedness < 0.35) | 0.15 | 0.30 |

Intervals are bootstrap percentiles over questions. Per-question answers are in
[`reports/eval_base_rag.json`](reports/eval_base_rag.json) and
[`reports/eval_finetuned_rag.json`](reports/eval_finetuned_rag.json).

**What the numbers say**

- Retrieval does the work: the gold document is in the top-4 chunks for 95% of answerable questions.
- **Fine-tuning did not help on this evaluation.** Recall is lower for the adapter than for the base
  model on both seen and unseen documents; the differences are within noise (the interval of the
  paired difference on seen questions is −0.19 to 0.00, and −0.18 to +0.10 on unseen), so the honest
  reading is "no measurable gain". Training data is the corpus text copied as the answer, which teaches
  the model to paraphrase documents, not to answer questions.
- **Abstention is the main weakness.** Both systems answer almost every unanswerable question with an
  invented figure (for example a fertilizer dose or a reservoir capacity) instead of saying the context
  does not contain it, even though the prompt asks for that.

## Running on Windows (notes)

This was developed and run on Windows 11 + RTX 3060. Two native-DLL ordering quirks and
their fixes (already applied in the code):

- Import `datasets`/`pyarrow` **before** `torch`, and load the embedding model **before**
  opening the `psycopg` connection — importing them in the reverse order triggers an
  OpenMP/`libpq` DLL clash that segfaults the process.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` before training.

On Linux (the production target) these quirks do not occur.

## Components

| Capability | Where in this repo |
|---|---|
| Spanish dataset pipeline: cleaning, dedup, tokenization | `src/data_pipeline/` |
| CPT/SFT on Llama/Qwen | `src/finetune/train_qlora.py` (Qwen2.5) |
| SFT with LoRA/QLoRA, HuggingFace + TRL | `train_qlora.py` (TRL `SFTTrainer` + PEFT) |
| Quantization for on-prem (GGUF, llama.cpp) | `docs/quantization.md` |
| RAG on PostgreSQL + pgvector | `src/rag/` + `docker-compose.yml` |
| Evaluation suite + hallucination monitoring | `src/eval/evaluate.py` |

---

## Limitations

- **Small, hand-written benchmark.** 100 questions over 59 short documents written by the author; the
  groups are small (25 and 20 questions), so the intervals above are wide. Keyword recall is a crude
  proxy (a correct answer phrased differently scores zero); semantic similarity and groundedness are
  embedding-based heuristics, not human judgements.
- **The questions are written from the same documents they test**, so this measures reading a known
  corpus, not general agronomy knowledge. It says nothing about questions outside that corpus.
- Single run, single seed, greedy decoding, a 0.5B model. The fine-tuning set is 45 examples; the
  conclusion that it did not help applies to this recipe and size only.
- Corpus facts are standard textbook values (FAO-56, FAO irrigation and drainage papers, usual
  hydrology formulas) summarised by the author, not an audited source; check them before real use.
- Unanswerable questions are not handled: see the abstention results above.

## Tech stack

Python · PyTorch · HuggingFace `transformers` / `datasets` · `TRL` · `PEFT` ·
`bitsandbytes` (4-bit) · `sentence-transformers` · PostgreSQL + `pgvector` · Docker.

## Repo layout

```
agrollm-es/
├── data/
│   ├── raw/corpus.jsonl          # 59-document Spanish domain corpus (16 flagged "heldout")
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
