# Unit 4 — Model Evaluation

Official chapter URL base: https://huggingface.co/learn/smol-course/unit4/
NotebookLM source: [`notebooklm/unit_4.txt`](../notebooklm/unit_4.txt)

## Chapters (in order)

- **Coming soon!** — https://huggingface.co/learn/smol-course/unit4/1

## Status

- [x] **Tracked.** The chapter content is a placeholder upstream — the only chapter `unit4/1.mdx` in `huggingface/smol-course` currently reads "Coming soon!".
- [ ] Re-run `_build_notebooklm.py` once `units/en/unit4/` gains real content; this folder's README + a practice quiz will follow then.
- [ ] No HF-site quiz exists for Unit 4 yet.

## What this unit *will* cover (per the original course schedule and the chapter's own framing)

Model evaluation for fine-tuned LLMs: benchmark-based eval (TruthfulQA,
GSM8K, MMLU, HellaSwag, ARC-Challenge), `lighteval` + `vllm` integration,
running eval as an HF Job, and pushing results to a Hub dataset for the
leaderboard. The Unit 2 / DPO submission flow already references this
pattern:

```bash
hf jobs uv run \
  --flavor a10g-large \
  --with "lighteval[vllm]" \
  --secrets HF_TOKEN \
  lighteval vllm "model_name=<user>/<model>" \
  "lighteval|truthfulqa:mc2|0|0,lighteval|hellaswag|0|0,lighteval|arc:challenge|0|0" \
  --push-to-hub --results-org <user>
```

When the upstream chapter lands, the workflow plan is:

1. `git -C _source pull && python _build_notebooklm.py` — refresh `notebooklm/unit_4.txt` + this README.
2. Eval the Unit 1 SFT adapter and the Unit 2 DPO adapter side-by-side on a small benchmark mix.
3. Push the results dataset and (optionally) update the smol-course leaderboard.
4. Draft `unit_4/practice_quiz.md` from the new chapters.
