# HuggingFace Smol Course (Fine-Tuning Language Models)

Working through the [HuggingFace Smol Course](https://huggingface.co/learn/smol-course/unit0/1) — fine-tuning language models with TRL, Transformers, and PEFT.

## Course Units

| Unit | Topic | Chapters |
|------|-------|----------|
| 0 | Welcome to the Fine-Tuning Course | 1 |
| 1 | Instruction Tuning (Chat templates, SFT, LoRA/PEFT, HF Jobs, final project) | 7 |
| 2 | Preference Alignment (DPO + advanced) | 4 |
| 3 | Vision Language Models | 4 |
| 4 | Model Evaluation | *Coming soon* |

(Units 5–7 — Reinforcement Learning, Synthetic Data, Award Ceremony — were on the original schedule but not yet released in `units/en/` upstream.)

## Certification

| Level | Requirement |
|-------|-------------|
| **Fundamentals** | Complete Unit 1 |
| **Completion** | Complete all released units **+** submit a final project |

## Structure

- `unit_*/` — per-unit work (hands-on artifacts, fine-tuned model links, notes).
- `notebooklm/unit_*.txt` — concatenated unit text from the official `.md`/`.mdx` source, formatted for [NotebookLM](https://notebooklm.google.com/) (audio overviews + flashcards).
- `_source/` — shallow clone of [`huggingface/smol-course`](https://github.com/huggingface/smol-course) used to build the notebooklm files. **Gitignored.**

## Setup

```bash
pip install -r requirements.txt
```

You'll also want:
- A **HuggingFace account** with `HF_TOKEN` set (for pushing fine-tuned models).
- **GPU access** — Hugging Face Pro / Colab / a local CUDA GPU. CPU-only is too slow for SFT/DPO.
