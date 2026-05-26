# Plan: Complete the Smol Course (Fundamentals → Completion Cert)

## Progress Tracker

**Overall: 2/5 fully done (Units 0, 1). Unit 2 DPO launched; Unit 3 VLM script + practice quiz ready and queued; Unit 4 placeholder tracked.**

| Done | Unit | Topic | Hands-on? | Chapters |
|:----:|------|-------|:---------:|---------:|
| ✅ | 0 | Welcome / Onboarding | — | 1 |
| ✅ | 1 | Instruction Tuning — SFT+LoRA done; adapter at [VoicesColeby/smollm3-3b-sft-lora](https://huggingface.co/VoicesColeby/smollm3-3b-sft-lora) (train loss 1.04→0.71, ~55 min); practice quiz drafted; HF-site quiz pending | ✅ done | 7 |
| 🟡 | 2 | Preference Alignment (DPO) — running on ultrafeedback_binarized → [VoicesColeby/smollm3-3b-dpo-lora](https://huggingface.co/VoicesColeby/smollm3-3b-dpo-lora); practice quiz drafted | ✅ launched | 4 |
| 🟡 | 3 | Vision Language Models — SmolVLM2-2.2B + ChartQA script ready and queued; practice quiz drafted | ✅ queued | 4 |
| ⏸ | 4 | Model Evaluation — *upstream is "Coming soon!"; tracked, will revisit on update* | n/a | 1 placeholder |

**Certs**
- **Fundamentals** = complete Unit 1.
- **Completion** = complete all released units + submit a final project.

---

## Workflow per unit

1. **Read** the official chapters online or via `notebooklm/unit_N.txt` (audio overviews + flashcards via NotebookLM).
2. **Fine-tune** the unit's target model end-to-end under `unit_N/<run-name>/` — push to the Hub with proper tags (re-use the `huggingface-model-publish` skill from the Context Course).
3. **Take the unit's quiz/assignment** on the HF course site if one exists.
4. **Mark done** here and update the unit's local README with the result + model link.

---

## Recommended order

### Unit 0 — Welcome
- One-page onboarding. Confirm HF account + `HF_TOKEN` + GPU access.

### Unit 1 — Instruction Tuning (cert-critical for Fundamentals)
- Read 7 chapters: intro → chat templates → SFT → LoRA/PEFT → practical exercises → HF Jobs → submit final project.
- **Fine-tune** a SmolLM (or similar small model) on an instruction dataset (`HuggingFaceTB/smoltalk` or your domain) using TRL's SFT + LoRA.
- Push tagged appropriately; record the link in `unit_1/README.md`.

### Unit 2 — Preference Alignment (DPO)
- Read 4 chapters.
- **DPO fine-tune** from the Unit 1 model using a preference dataset (e.g. `argilla/distilabel-intel-orca-dpo-pairs` or similar).
- Push and record.

### Unit 3 — Vision Language Models
- Read 4 chapters.
- **VLM fine-tune** (small VLM like SmolVLM) on a multimodal task; push.

### Unit 4 — Model Evaluation
- Currently a placeholder ("Coming soon!") in `units/en/unit4/1.mdx`. Re-run `_build_notebooklm.py` after the upstream lands content.

### Final Project (for Completion cert)
- Pick a task; combine SFT + DPO (+ optional VLM); push the model; write up.

---

## Verification

Track progress on your HuggingFace profile (models pushed) and the course
site's progress tracker (when it's published / once Unit 1 quiz exists).
