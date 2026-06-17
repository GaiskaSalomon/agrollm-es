# Quantization & on-premise deployment (GGUF · llama.cpp · Ollama)

After QLoRA fine-tuning you have a LoRA **adapter**. To deploy on-prem on CPU or a
single GPU you typically: merge the adapter into the base, convert to **GGUF**, and
quantize. This is the "model quantization for on-premise deployment" step.

## 1. Merge LoRA adapter into the base

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "outputs/qwen-agro-lora")
model = model.merge_and_unload()                 # fold adapter into weights
model.save_pretrained("outputs/qwen-agro-merged")
AutoTokenizer.from_pretrained("outputs/qwen-agro-lora").save_pretrained("outputs/qwen-agro-merged")
```

## 2. Convert to GGUF and quantize with llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
pip install -r requirements.txt

# HF -> GGUF (fp16)
python convert_hf_to_gguf.py ../outputs/qwen-agro-merged \
       --outfile qwen-agro-f16.gguf --outtype f16

# Quantize to 4-bit (Q4_K_M is a good size/quality tradeoff for on-prem)
./llama-quantize qwen-agro-f16.gguf qwen-agro-q4_k_m.gguf Q4_K_M
```

## 3. Serve on-prem

**llama.cpp server:**
```bash
./llama-server -m qwen-agro-q4_k_m.gguf -c 4096 --host 0.0.0.0 --port 8080
```

**Ollama:**
```bash
# Modelfile
printf 'FROM ./qwen-agro-q4_k_m.gguf\nPARAMETER temperature 0.2\n' > Modelfile
ollama create qwen-agro -f Modelfile
ollama run qwen-agro "¿Cómo se calcula la lámina de riego?"
```

**Apple Silicon (MLX):** convert with `mlx_lm.convert -q --hf-path outputs/qwen-agro-merged`
to run quantized on Mac hardware.

## Quantization tradeoffs

| Format   | Bits | Size (0.5B) | Use case                         |
|----------|------|-------------|----------------------------------|
| f16      | 16   | ~1.0 GB     | reference / further conversion   |
| Q8_0     | 8    | ~0.5 GB     | highest quality quantized        |
| Q4_K_M   | ~4.5 | ~0.4 GB     | recommended CPU/edge default     |
| Q4_0     | 4    | ~0.3 GB     | smallest, some quality loss      |

Point the RAG generator at the served endpoint to complete the on-prem stack.
