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
- [x] **Hands-on fine-tune done.** Adapter pushed to: **[VoicesColeby/smollm3-3b-sft-lora](https://huggingface.co/VoicesColeby/smollm3-3b-sft-lora)** (`pipeline_tag: text-generation`, tagged `trl`, `sft`, base `HuggingFaceTB/SmolLM3-3B-Base`).
- [x] Practice quiz drafted — [`practice_quiz.md`](practice_quiz.md)
- [ ] Quiz on the HF site — `https://huggingface.co/spaces/smol-course/unit_1_quiz`

## The fine-tune — `scripts/train_smollm3_sft_lora.py`

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM3-3B-Base` |
| Dataset | `HuggingFaceTB/smoltalk2_everyday_convs_think` — 1,851 rows, single `text` field, already SmolLM3-chat-templated |
| Method | TRL `SFTTrainer` + PEFT LoRA (`r=16, α=32`, dropout 0.05; target_modules q/k/v/o_proj + gate/up/down_proj) + 4-bit NF4 quant via bitsandbytes |
| Hyperparams | lr=5e-5 cosine, warmup=100, batch=2 × grad-accum=8 (effective 16), `max_steps=200`, `max_length=1024`, bf16, gradient checkpointing (non-reentrant), `paged_adamw_8bit` |
| Hardware | RTX 5060 Ti / 16 GB |
| Wall-clock | **55:27** (3327 s) |
| Loss | started ~1.04, decreased smoothly to ~0.71 by step 200 |
| Mean token accuracy | ~0.78 at end |
| LoRA adapter size | 60.5 MB |

### Why the recipe deviates from the chapter's defaults

Chapter 3's recipe uses `max_length=2048, max_steps=1000`. On the local 16 GB
GPU at `max_length=2048`, observed step time was ~16–72 s and the ETA
projected ~12–20 hours. Trimming to `max_length=1024` (most samples in this
dataset fit; longer get truncated) brought step time down to a stable
~16.5 s, giving ~55 min for 200 steps — still ~1.7 epochs through 1851 rows
at effective batch 16, plenty of signal to demonstrate SFT.

If you have more VRAM (24 GB+) or are running on HF Jobs, raise both:

```bash
SMOL_MAX_LEN=2048 SMOL_MAX_STEPS=1000 python unit_1/scripts/train_smollm3_sft_lora.py
```

## Three Windows-specific gotchas worth flagging

1. **`PYTHONUTF8=1` is required.** TRL 1.5.0's `chat_template_utils.py` reads `deepseekv3.jinja` without specifying `encoding`, so Python's default cp1252 on Windows raises `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. Workaround: launch with `PYTHONUTF8=1`. Upstream-PR candidate on `huggingface/trl` (one-line fix: `read_text(encoding="utf-8")`).
2. **PEFT 0.18+ requires explicit `target_modules`.** It no longer auto-infers them for unmapped architectures (`ValueError: Please specify target_modules or target_parameters in peft_config`). The Llama/SmolLM3-style attention + MLP linears are spelled out in `LoraConfig` in the script.
3. **bitsandbytes works on Windows** — `0.49.2` ships a Windows wheel; verified 4-bit Linear forward on the RTX 5060 Ti before launching.

## One subtle pitfall I hit (fixed in the script)

`trainer.push_to_hub(...)` defaults to **`output_dir basename`** under the
current user when `hub_model_id` isn't set on `SFTConfig`. I now set
`hub_model_id=HUB_REPO` explicitly. The first run landed at
`VoicesColeby/smollm3-3b-sft-lora` (the output-dir name), which is fine —
just call it out so you don't lose track of where your adapter actually went.

## Run it yourself

```bash
# Windows
set PYTHONUTF8=1
python unit_1/scripts/train_smollm3_sft_lora.py

# macOS / Linux
python unit_1/scripts/train_smollm3_sft_lora.py
```

Env vars you can override: `SMOL_MODEL`, `SMOL_DATASET`, `SMOL_OUT`,
`SMOL_HUB_REPO`, `SMOL_MAX_STEPS`, `SMOL_MAX_LEN`.

## Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B-Base", dtype=torch.bfloat16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
model = PeftModel.from_pretrained(base, "VoicesColeby/smollm3-3b-sft-lora")

messages = [{"role": "user", "content": "Explain LoRA in one paragraph."}]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
```
