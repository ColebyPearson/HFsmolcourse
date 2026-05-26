# Unit 3 — Vision Language Models

Official chapter URL base: https://huggingface.co/learn/smol-course/unit3/
NotebookLM source: [`notebooklm/unit_3.txt`](../notebooklm/unit_3.txt)

## Chapters (in order)

- **Introduction to Vision Language Models** — https://huggingface.co/learn/smol-course/unit3/1
- **Using Pretrained VLMs** — https://huggingface.co/learn/smol-course/unit3/2
- **Fine-Tuning VLMs** — https://huggingface.co/learn/smol-course/unit3/3
- **Hands-On Fine-Tuning VLMs** — https://huggingface.co/learn/smol-course/unit3/4

## Status

- [x] Read all chapters (or listen via NotebookLM)
- [x] **VLM SFT script ready** — [`scripts/train_smolvlm_sft_lora.py`](scripts/train_smolvlm_sft_lora.py). Will publish to **[VoicesColeby/smolvlm2-2.2b-sft-lora-chartqa](https://huggingface.co/VoicesColeby/smolvlm2-2.2b-sft-lora-chartqa)** when run.
- [x] Practice quiz drafted — [`practice_quiz.md`](practice_quiz.md)
- [ ] Train (queued — launches when Unit 2 DPO finishes and the GPU is free)
- [ ] Quiz on the HF site — `https://huggingface.co/spaces/smol-course/unit_4_quiz` (course numbering)

## The VLM fine-tune — `scripts/train_smolvlm_sft_lora.py`

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (loaded via `AutoModelForImageTextToText` + `AutoProcessor`) |
| Dataset | `HuggingFaceM4/ChartQA` (10% slice of `train`) — chart visual QA |
| Method | TRL `SFTTrainer` + PEFT LoRA, bf16 (no 4-bit — smaller model + vision tower is finicky with bnb) |
| LoRA | `r=8, α=16, dropout=0.05, target_modules=["q_proj", "v_proj"]` (per the chapter recipe — narrower than the LLM Unit 1 set) |
| Hyperparams | lr=1e-4 cosine, warmup=20, batch=1 × grad-accum=4 (effective 4), `max_steps=150`, **`max_length=None`** (critical: otherwise image tokens get truncated) |
| Data shape | `{"images": [PIL.Image], "messages": [{role, content:[{type:image|text, …}, …]}, …]}` with a system message defining the chart-analyst role |
| Hardware | RTX 5060 Ti, 16 GB |

### The four VLM-specific things to remember

1. **`AutoModelForImageTextToText`**, not `AutoModelForCausalLM`. Different head.
2. **`AutoProcessor`**, not `AutoTokenizer`. Wraps text tokenizer + image processor.
3. **`max_length=None`** in `SFTConfig`. The chapter explicitly warns: truncation removes image tokens and silently breaks training.
4. **Image-aware message content** — each message's `content` is a *list* of typed parts: `{"type": "image", "image": img}` and `{"type": "text", "text": "..."}`.

### Why fewer LoRA target modules than Unit 1

The course recipe targets only `q_proj` and `v_proj`. VLMs have both a vision
tower and a language head; widening LoRA to all attention+MLP linears
(`q/k/v/o + gate/up/down`) inflates trainable params without much benefit on
a small dataset slice. Narrow LoRA also keeps the adapter under 20 MB.

## Run it yourself

```bash
# Windows
set PYTHONUTF8=1
python unit_3/scripts/train_smolvlm_sft_lora.py

# macOS / Linux
python unit_3/scripts/train_smolvlm_sft_lora.py
```

Env vars: `SMOL_MODEL`, `SMOL_DATASET`, `SMOL_OUT`, `SMOL_HUB_REPO`, `SMOL_MAX_STEPS`.
