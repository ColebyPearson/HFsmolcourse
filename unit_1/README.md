# Unit 1 — Instruction Tuning

Official chapter URL base: https://huggingface.co/learn/smol-course/unit1/
NotebookLM source: [`notebooklm/unit_1.txt`](../notebooklm/unit_1.txt)

## Chapters (in order)

- **Introduction to Instruction Tuning** — https://huggingface.co/learn/smol-course/unit1/1
- **Chat Templates** — https://huggingface.co/learn/smol-course/unit1/2
- **Supervised Fine-Tuning** — https://huggingface.co/learn/smol-course/unit1/3
- **LoRA and PEFT** — https://huggingface.co/learn/smol-course/unit1/3a
- **Practical Exercises** — https://huggingface.co/learn/smol-course/unit1/4
- **Training with Hugging Face Jobs** — https://huggingface.co/learn/smol-course/unit1/5
- **Submit your final project!** — https://huggingface.co/learn/smol-course/unit1/6

## Status

- [x] Read all chapters (or listen via NotebookLM)
- [x] **Hands-on fine-tune** — script written and **launched in background**: [`scripts/train_smollm3_sft_lora.py`](scripts/train_smollm3_sft_lora.py). Hub repo (will appear when training pushes): [`VoicesColeby/smollm3-3b-sft-lora-smoltalk-everyday`](https://huggingface.co/VoicesColeby/smollm3-3b-sft-lora-smoltalk-everyday).
- [x] Practice quiz drafted — [`practice_quiz.md`](practice_quiz.md)
- [ ] Quiz on the HF site — `https://huggingface.co/spaces/smol-course/unit_1_quiz`

## The fine-tune — `scripts/train_smollm3_sft_lora.py`

Implements the course's recipe verbatim:

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM3-3B-Base` |
| Dataset | `HuggingFaceTB/smoltalk2_everyday_convs_think` (1,851 rows, single `text` field, already SmolLM3-chat-templated) |
| Method | TRL `SFTTrainer` + PEFT LoRA (`r=16`, `α=32`, dropout 0.05) + 4-bit NF4 quant via bitsandbytes |
| Hyperparams | lr=5e-5, warmup=100, batch=2 × grad-accum=8 (effective 16), max_steps=1000, max_length=2048, bf16, gradient checkpointing (non-reentrant), cosine LR, `paged_adamw_8bit` |
| Local cost | RTX 5060 Ti / 16 GB — fits via 4-bit + LoRA |

The script gates the Hub push on training success (no checkpoint-time push).

## Two Windows-specific gotchas worth flagging

1. **`PYTHONUTF8=1` is required.** TRL 1.5.0's `chat_template_utils.py` reads `deepseekv3.jinja` without specifying `encoding`, so Python's default cp1252 on Windows raises `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. Setting `PYTHONUTF8=1` before launching fixes it. Worth an upstream issue/PR on `huggingface/trl` (one-line fix: `read_text(encoding="utf-8")`). Repro: tried to import `SFTTrainer` on Windows + TRL 1.5.0 → fails with the cp1252 decode error.
2. **bitsandbytes works on Windows now** — `0.49.2` ships a Windows wheel; verified 4-bit Linear forward on the RTX 5060 Ti before launching training.

## Run it yourself

```bash
# Windows
set PYTHONUTF8=1
python unit_1/scripts/train_smollm3_sft_lora.py

# macOS / Linux (Python defaults to UTF-8, no env var needed)
python unit_1/scripts/train_smollm3_sft_lora.py
```

Env vars you can override: `SMOL_MODEL`, `SMOL_DATASET`, `SMOL_OUT`, `SMOL_HUB_REPO`, `SMOL_MAX_STEPS`, `SMOL_MAX_LEN`.
