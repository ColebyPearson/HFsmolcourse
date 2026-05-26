# HuggingFace Smol Course (Fine-Tuning LLMs)

## Project Overview
This repo contains hands-on work for the HuggingFace Smol Course — fine-tuning language models with TRL, Transformers, and PEFT.
Course URL: https://huggingface.co/learn/smol-course/unit0/1
GitHub source: https://github.com/huggingface/smol-course

## Certification
- **Fundamentals**: complete Unit 1 (Instruction Tuning).
- **Completion**: complete all released units (1–4 currently) + submit a final project.

## Hands-on / fine-tuning targets
- **Unit 1** — SFT a small model on an instruction dataset (chat templates, LoRA, optionally HF Jobs).
- **Unit 2** — DPO fine-tuning for preference alignment.
- **Unit 3** — VLM fine-tuning on a multimodal dataset.
- **Unit 4** — *Coming soon* upstream.

Final project: pick a task, fine-tune end-to-end, push to the Hub, write up.

## Structure
- `unit_*/` — per-unit artifacts and notes; fine-tuned model links land here.
- `notebooklm/unit_*.txt` — concatenated `.md`/`.mdx` per unit (NotebookLM-ready).
- `_source/` — local clone of `huggingface/smol-course` (gitignored; used to build `notebooklm/`).
- `_build_notebooklm.py` — re-runnable: refreshes `notebooklm/` + per-unit READMEs from the upstream toctree.

## Key libraries
- `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`
- `evaluate` (Unit 4 when it lands)
- `wandb` optional for run tracking

## Reference hardware
This box has a CUDA GPU; SFT/DPO/VLM fine-tuning runs locally. We learned to
load datasets via the auto-converted parquet branches (datasets 4.x dropped
script loaders) and to decode audio/image bytes manually when torchcodec/
ffmpeg are unavailable — same lessons as the HF Audio and Context courses.
