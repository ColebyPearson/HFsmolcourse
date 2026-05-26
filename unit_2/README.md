# Unit 2 — Preference Alignment (DPO)

Official chapter URL base: https://huggingface.co/learn/smol-course/unit2/
NotebookLM source: [`notebooklm/unit_2.txt`](../notebooklm/unit_2.txt)

## Chapters (in order)

- **Introduction to Preference Alignment** — https://huggingface.co/learn/smol-course/unit2/1
- **Direct Preference Optimization (DPO)** — https://huggingface.co/learn/smol-course/unit2/2
- **Advanced DPO Techniques** — https://huggingface.co/learn/smol-course/unit2/3
- **DPO Hands-on Implementation** — https://huggingface.co/learn/smol-course/unit2/4

## Status

- [x] Read all chapters (or listen via NotebookLM)
- [x] **Hands-on DPO fine-tune — launched** ([`scripts/train_smollm3_dpo_lora.py`](scripts/train_smollm3_dpo_lora.py)). Adapter will publish to **[VoicesColeby/smollm3-3b-dpo-lora](https://huggingface.co/VoicesColeby/smollm3-3b-dpo-lora)** on completion.
- [x] Practice quiz drafted — [`practice_quiz.md`](practice_quiz.md)
- [ ] Quiz on the HF site — `https://huggingface.co/spaces/smol-course/unit_3_quiz` (course numbering)

## The DPO fine-tune — `scripts/train_smollm3_dpo_lora.py`

| Field | Value |
|---|---|
| Starting policy (= `π_ref`) | `HuggingFaceTB/SmolLM3-3B` (the already-SFT'd instruction-tuned variant — the course's default for DPO) |
| Preference dataset | `trl-lib/ultrafeedback_binarized` (canonical TRL preference format) |
| Method | TRL `DPOTrainer` + PEFT LoRA + 4-bit NF4 quant |
| Reference policy | Reused base weights via the LoRA-disabling trick — no separate `ref_model` |
| Hyperparams | `beta=0.1`, lr=5e-7 cosine, warmup=50, batch=1 × grad-accum=8 (effective 8 — DPO sees chosen+rejected per step), `max_steps=200`, `max_prompt_length=512`, `max_length=1024`, bf16, gradient checkpointing |
| Hardware | RTX 5060 Ti, 16 GB |
| Hub model id | Set via `DPOConfig(hub_model_id=…)` (lesson learned from Unit 1) |

### Deviation from the course recipe

The course's CLI example uses `--max_steps 1000`. We cap at 200 here for the
same reason as Unit 1: the 16 GB GPU's wall-clock budget. Same DPO mechanics,
fewer optimiser steps, still ample preference signal. To run the chapter's
full recipe, set `SMOL_MAX_STEPS=1000` (and ideally move to a 24 GB+ GPU or
HF Jobs).

### An alternative starting point

Instead of HF's instruction-tuned `SmolLM3-3B`, you can start DPO from the
Unit 1 SFT adapter on `SmolLM3-3B-Base` for a "build on what you trained"
story. Add the loading sequence

```python
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base", ...)
model = PeftModel.from_pretrained(model, "VoicesColeby/smollm3-3b-sft-lora").merge_and_unload()
```

before instantiating `DPOTrainer`. The published Unit 2 model in this repo
follows the simpler course-canonical path so it's directly comparable to the
chapter's example.

## Run it yourself

```bash
# Windows
set PYTHONUTF8=1
python unit_2/scripts/train_smollm3_dpo_lora.py

# macOS / Linux
python unit_2/scripts/train_smollm3_dpo_lora.py
```

Env vars: `SMOL_MODEL`, `SMOL_DATASET`, `SMOL_OUT`, `SMOL_HUB_REPO`,
`SMOL_MAX_STEPS`, `SMOL_MAX_LEN`, `SMOL_MAX_PROMPT_LEN`, `SMOL_BETA`.
