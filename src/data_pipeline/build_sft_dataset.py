"""Turn a cleaned corpus into a supervised fine-tuning (SFT) dataset.

Produces conversational JSONL compatible with TRL's `SFTTrainer`:
    {"messages": [{"role": "system", ...},
                  {"role": "user", ...},
                  {"role": "assistant", ...}]}

Two sources of examples:
  1. Curated, hand-written QA pairs (high quality, domain-correct anchors).
  2. Synthetic instruction pairs generated from each corpus document via templates
     (the document text becomes a grounded answer to a generated question).

In production you would replace (2) with a teacher LLM that drafts and a domain
expert that reviews — that is exactly the RLHF/DPO annotation loop. Here it is kept
deterministic so the dataset builds offline.

Usage:
    python -m src.data_pipeline.build_sft_dataset --in data/interim/dedup.jsonl \
                                                  --out data/processed/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "Eres un asistente técnico experto en agronomía, riego e hidrología. "
    "Respondes en español de forma precisa, concisa y basada en evidencia."
)

# Question templates keyed by a topic keyword found in the document.
TEMPLATES = {
    "lámina de riego": "¿Cómo se calcula la lámina de riego de un cultivo?",
    "método racional": "¿Qué es el método racional y cuándo se aplica?",
    "eficiencia": "¿Qué eficiencia tienen los distintos métodos de riego?",
    "evapotranspiración de referencia": "¿Cómo se calcula la evapotranspiración de referencia ET0?",
    "cosecha de agua": "¿Cómo funciona la cosecha de agua de lluvia para consumo doméstico?",
    "coeficiente de escurrimiento": "¿De qué depende el coeficiente de escurrimiento C?",
    "salinidad": "¿Cómo se interpreta la salinidad del agua de riego?",
    "periodo de retorno": "¿Qué es el periodo de retorno y qué valores se usan en diseño?",
    "riego deficitario": "¿En qué consiste el riego deficitario controlado?",
    "balance hídrico": "¿Qué es el balance hídrico de una cuenca?",
    "capacidad de campo": "¿Qué es la capacidad de campo y para qué sirve?",
    "número de curva": "¿Qué representa el número de curva CN?",
}

GENERIC_QUESTIONS = [
    "Explica el siguiente concepto técnico.",
    "Resume de forma clara la siguiente información.",
    "¿Qué dice la teoría sobre este tema?",
]


def make_question(text: str) -> str:
    low = text.lower()
    for kw, q in TEMPLATES.items():
        if kw in low:
            return q
    return random.choice(GENERIC_QUESTIONS)


# Hand-written anchors (kept verbatim, domain-validated).
CURATED = [
    {
        "user": "Si la ET0 es de 6 mm/día y el Kc del maíz en etapa media es 1.15, "
                "¿cuál es la ETc diaria?",
        "assistant": "La evapotranspiración del cultivo se obtiene como ETc = ET0 × Kc. "
                     "Con ET0 = 6 mm/día y Kc = 1.15, la ETc = 6 × 1.15 = 6.9 mm/día.",
    },
    {
        "user": "Tengo una cuenca de 50 hectáreas, C = 0.3 e intensidad de 40 mm/h. "
                "¿Cuál es el gasto pico por el método racional?",
        "assistant": "Con el método racional Q = C·i·A/360 = (0.3 × 40 × 50)/360 ≈ 1.67 m³/s. "
                     "Recuerda que el método es válido para cuencas pequeñas (< 25 km²).",
    },
]


def run(in_path: Path, out_path: Path, seed: int = 13) -> int:
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = []

    # Synthetic from corpus.
    for line in in_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec["text"]
        question = make_question(text)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": text},
                ]
            }
        )

    # Curated anchors.
    for qa in CURATED:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qa["user"]},
                    {"role": "assistant", "content": qa["assistant"]},
                ]
            }
        )

    random.shuffle(examples)
    with out_path.open("w", encoding="utf-8") as fout:
        for ex in examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an SFT dataset from a corpus.")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    args = ap.parse_args()

    n = run(args.in_path, args.out_path)
    print(f"[build_sft] wrote {n} SFT examples -> {args.out_path}")


if __name__ == "__main__":
    main()
