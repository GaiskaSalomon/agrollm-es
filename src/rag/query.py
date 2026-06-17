"""RAG query: retrieve grounded context from pgvector, then generate an answer.

Generation backends (auto-detected, in order):
  1. Fine-tuned local model in outputs/ (base + LoRA adapter), if transformers+torch
     are available.
  2. Otherwise, "retrieval-only" mode: return the top passages as the grounded answer.

Usage:
    python -m src.rag.query --q "¿Cómo se calcula la lámina de riego para maíz?"
    python -m src.rag.query --q "..." --adapter outputs/qwen-agro-lora
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.rag import db
from src.rag.embeddings import embed

PROMPT_TEMPLATE = (
    "Eres un asistente técnico en agronomía e hidrología. Responde la pregunta usando "
    "EXCLUSIVAMENTE el contexto. Si el contexto no contiene la respuesta, dilo "
    "explícitamente.\n\n### Contexto:\n{context}\n\n### Pregunta:\n{question}\n\n### Respuesta:\n"
)


def retrieve(question: str, k: int = 4):
    # Embed first (loads torch), then connect — avoids a torch/libpq DLL clash on Windows.
    qvec = embed([question])[0]
    conn = db.get_conn()
    return db.search(conn, qvec, k=k)


def generate_with_model(prompt: str, base: str, adapter: str | None) -> str | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None

    src = adapter if adapter and Path(adapter).exists() else base
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float32, device_map="auto" if torch.cuda.is_available() else None
    )
    if adapter and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def answer(question: str, base: str, adapter: str | None, k: int = 4) -> dict:
    hits = retrieve(question, k=k)
    context = "\n".join(f"- {content}" for content, _src, _dist in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    generated = generate_with_model(prompt, base, adapter)
    mode = "generative"
    if generated is None:
        generated = context  # retrieval-only fallback
        mode = "retrieval-only"

    return {
        "question": question,
        "answer": generated,
        "mode": mode,
        "sources": [{"source": s, "distance": round(float(d), 4)} for _c, s, d in hits],
        "context": [c for c, _s, _d in hits],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG query over pgvector.")
    ap.add_argument("--q", required=True, help="Question in Spanish.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen-agro-lora")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    res = answer(args.q, args.base, args.adapter, k=args.k)
    print(f"\n[{res['mode']}] {res['question']}\n")
    print(res["answer"])
    print("\nFuentes:", res["sources"])


if __name__ == "__main__":
    main()
