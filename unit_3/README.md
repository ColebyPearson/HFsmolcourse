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
- [x] **VLM SFT done.** Adapter pushed: **[VoicesColeby/smolvlm2-2.2b-sft-lora-chartqa](https://huggingface.co/VoicesColeby/smolvlm2-2.2b-sft-lora-chartqa)** (`pipeline_tag: image-text-to-text`, tagged `sft`, `trl`, `generated_from_trainer`). Wall-clock **21:15**, 150 steps; running train loss **0.51 → 0.33**; **mean-token-accuracy ~0.94** by end (chart-VQA answers are short — `"3"`, `"60%"` — so high token accuracy is the expected good signal).
- [x] Practice quiz drafted — [`practice_quiz.md`](practice_quiz.md)
- [ ] Quiz on the HF site — `https://huggingface.co/spaces/smol-course/unit_4_quiz` (course numbering)

## What the chapter's example needs to update (upstream candidate)

`units/en/unit3/4.md`'s `format_data()` writes messages in **typed-parts** form:

```python
{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]}
```

Current TRL's SFT collator (`trl.data_utils.prepare_multimodal_messages`)
counts `<image>` tokens in **string** content and ignores typed-parts — running
the chapter's recipe verbatim raises
`ValueError: Number of images provided (1) does not match number of image placeholders (0)`.
The working format (what this script uses) is plain-string content with no
explicit `<image>`; TRL injects the placeholder itself. Two upstream
candidates here: a docs PR on `huggingface/smol-course` and possibly a TRL
improvement to accept typed-parts.

## The VLM fine-tune — `scripts/train_smolvlm_sft_lora.py`

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (loaded via `AutoModelForImageTextToText` + `AutoProcessor`) |
| Dataset | `HuggingFaceM4/ChartQA` (10% slice of `train`) — chart visual QA |
| Method | TRL `SFTTrainer` + PEFT LoRA, bf16 (no 4-bit — smaller model + vision tower is finicky with bnb) |
| LoRA | `r=8, α=16, dropout=0.05, target_modules=["q_proj", "v_proj"]` (per the chapter recipe — narrower than the LLM Unit 1 set) |
| Hyperparams | lr=1e-4 cosine, warmup=20, batch=1 × grad-accum=4 (effective 4), `max_steps=150`, **`max_length=None`** (critical: otherwise image tokens get truncated) |
| Data shape | `{"images": [PIL.Image], "messages": [{role, content: "plain text"}, ...]}` — plain-string content; TRL's collator auto-injects the `<image>` placeholder (don't add your own typed-parts or explicit placeholder) |
| Hardware | RTX 5060 Ti, 16 GB |
| Wall-clock | **21:15** (1,275 s) |

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
